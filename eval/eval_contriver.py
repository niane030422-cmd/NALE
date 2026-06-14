import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import re
import argparse
import json
import random
import unicodedata
import string
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from transformers import AutoTokenizer

from src.contriever import Contriever, M3contriver
from public_repo import repo_path


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def split_into_sentence(text: str) -> List[str]:
    sentence_endings = r'[。！？?.!؟；;]'
    sentences = re.split(fr'(?<=({sentence_endings}))\s*', text)
    sentences = [sent.strip() for sent in sentences if sent and sent.strip()]
    return sentences


def normalize_text(text: str) -> str:
    original_text = text

    text2 = text.translate(str.maketrans('', '', string.punctuation)).replace('  ', ' ').strip()
    if text2 != '' and not text2.isspace():
        text = text2

    text2 = unicodedata.normalize('NFKD', text)
    if text2 != '' and not text2.isspace():
        text = text2

    return text.lower().strip()


def build_chunks(document: str, tokenizer, lang: str, chunk_size: int = 200) -> List[str]:
    
    sentences = split_into_sentence(document)
    chunks = []
    cur = ""
    cur_tok = 0

    for sent in sentences:
        toks = tokenizer.encode(sent, add_special_tokens=False)
        if cur_tok + len(toks) <= chunk_size:
            if lang in ["zh", "ja", "ko"]:
                cur += sent
            else:
                cur += sent + " "
            cur_tok += len(toks)
        else:
            if cur.strip():
                chunks.append(cur.strip())
            cur = sent
            cur_tok = len(toks)

    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def sort_documents_by_similarity(contriever, con_tok, query: str, documents: List[str]) -> List[int]:
    """
    
    支持：
      - contriever + tokenizer (HF encoder)
      - contriever callable (M3contriver)
      - BM25
    """
    if con_tok is not None and contriever is not None:
        sentences = [query] + documents
        inputs = con_tok(sentences, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            embeddings = contriever(**inputs)
        q_emb = embeddings[0]              # [dim]
        d_emb = embeddings[1:]             # [n, dim]
        scores = torch.matmul(d_emb, q_emb)  # [n]
        scores = scores.detach().cpu()

    elif contriever is not None and con_tok is None:
        scores = contriever(query, documents)          # numpy
        scores = torch.from_numpy(scores)

    else:
        from rank_bm25 import BM25Okapi
        tokenized_corpus = [doc.split(" ") for doc in documents]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.split(" ")
        scores = torch.tensor(bm25.get_scores(tokenized_query), dtype=torch.float)

    sorted_idx = torch.argsort(scores, descending=True).cpu().numpy().tolist()
    return sorted_idx


# -------------------------
# gold chunk 标注 & recall@k
# -------------------------
def find_gold_chunk_ids(chunks: List[str], answers: List[str]) -> List[int]:
    """
    gold 定义：chunk 文本包含任一答案字符串（做 normalize 后包含匹配）
    answers 可多值（列表）
    """
    if not answers:
        return []

    answers_norm = [normalize_text(a) for a in answers if str(a).strip()]
    gold_ids = []
    for i, ch in enumerate(chunks):
        ch_norm = normalize_text(ch)
        for a in answers_norm:
            if a and a in ch_norm:
                gold_ids.append(i)
                break
    return gold_ids


def recall_at_k(ranked_ids: List[int], gold_ids: List[int], k: int) -> float:
    """Hit/Recall@K：Top-K 命中任一 gold 记为 1，否则 0"""
    if k <= 0:
        return 0.0
    if not gold_ids:
        return 0.0
    topk = set(ranked_ids[:k])
    return 1.0 if any(gid in topk for gid in gold_ids) else 0.0


# -------------------------
# 主流程：读数据 -> 切 chunk -> 检索 -> 计算 recall@k
# -------------------------


def ensure_list_answer(ans) -> List[str]:
    if ans is None:
        return []
    if isinstance(ans, list):
        return [str(x) for x in ans]
    s = str(ans)
    # 兼容你数据里 ", " 分隔多答案
    if ", " in s:
        return [x.strip() for x in s.split(", ") if x.strip()]
    return [s.strip()] if s.strip() else []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="json dataset path")
    parser.add_argument("--language", type=str, default="en", choices=["en","zh","ja","fr","ar","ko","es","pt"])
    parser.add_argument("--doc_lang", type=str, default="en", choices=["en","zh","ja","fr","ar","ko","es","pt"])
    parser.add_argument("--contriever_path", type=str, required=True, help="contriever path or bm25")
    parser.add_argument("--chunk_size", type=int, default=200, help="tokens per chunk")
    parser.add_argument("--max_k", type=int, default=20, help="compute recall@k for k=1..max_k")
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--save_path", type=str, default="./results/retriever_recall.jsonl")
    parser.add_argument("--skip_no_gold", action="store_true",
                        help="if set: skip instances where gold chunk can't be found")
    args = parser.parse_args()

    set_seed(args.seed)

    # 选择检索器
    contriever_name = ""
    contriever = None
    con_tok = None

    if "bm25" in args.contriever_path.lower():
        contriever_name = "bm25"
        contriever = None
        con_tok = None
    elif "mcontriever-msmarco" in args.contriever_path.lower():
        contriever_name = "mcontriever-msmarco"
        contriever = Contriever.from_pretrained(args.contriever_path)
        con_tok = AutoTokenizer.from_pretrained(args.contriever_path)
    else:
        contriever_name = "m3contriever"
        contriever = M3contriver(args.contriever_path)
        con_tok = None

    # 用一个通用 tokenizer
    chunk_tokenizer = AutoTokenizer.from_pretrained(repo_path("models", "facebook", "mcontriever-msmarco"))

    doc_id = 0
    instances = []
    noise_pool = {}
    with open(f'{args.dataset}','r', encoding='utf8') as f:
        json_data = json.load(f)
        eval_data = json_data[args.language]
        eval_doc_lang = json_data[args.doc_lang]

        for instance, doc_lang_line in zip(eval_data, eval_doc_lang): 
            instance["doc"] = doc_lang_line["doc"]
            instance["doc_lang_ans"] = doc_lang_line["QA"]["A"]

            instance["doc_lang"] = args.doc_lang
            instance["ori_lang_query"] = doc_lang_line["QA"]["Q"]

            if ', ' in instance["doc_lang_ans"]:
                instance["doc_lang_ans"] = instance["doc_lang_ans"].split(', ')
            noise_pool[doc_id] = doc_lang_line["doc"]
            doc_id += 1
            instances.append(instance)

    # 累计 recall@k
    recall_sums = {k: 0.0 for k in range(1, args.max_k + 1)}
    used = 0
    skipped = 0

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    with open(args.save_path, "w", encoding="utf8") as fw:
        for ins in instances:
            query = ins["QA"]["Q"].replace("،", ",")
            # gold 用 doc_lang_ans（文档语言答案）更稳
            gold_answers = ensure_list_answer(ins.get("doc_lang_ans"))

            doc = ins["doc"]
            lang = ins["doc_lang"]

            chunks = build_chunks(doc, chunk_tokenizer, lang=lang, chunk_size=args.chunk_size)
            gold_ids = find_gold_chunk_ids(chunks, gold_answers)

            if not gold_ids:
                if args.skip_no_gold:
                    skipped += 1
                    continue
                # 不跳过则按全 0 计
            ranked_ids = sort_documents_by_similarity(contriever, con_tok, query, chunks)

            per_item = {
                "id": ins.get("id", None),
                "query": query,
                "ori_lang_query": ins["ori_lang_query"],
                "doc_lang": lang,
                "contriever": contriever_name,
                "gold_answers": gold_answers,
                "gold_chunk_ids": gold_ids,
                "top_chunks": [chunks[i] for i in ranked_ids[:min(args.max_k, len(chunks))]],
            }

            item_recalls = {}
            for k in range(1, args.max_k + 1):
                r = recall_at_k(ranked_ids, gold_ids, k)
                recall_sums[k] += r
                item_recalls[f"recall@{k}"] = r

            per_item.update(item_recalls)
            fw.write(json.dumps(per_item, ensure_ascii=False) + "\n")
            used += 1

    # 汇总结果
    denom = max(1, used)  # 防止除 0
    summary = {
        "dataset": args.dataset,
        "query_lang": args.language,
        "doc_lang": args.doc_lang,
        "contriever": contriever_name,
        "chunk_size_tokens": args.chunk_size,
        "max_k": args.max_k,
        "seed": args.seed,
        "used_instances": used,
        "skipped_no_gold": skipped,
        "recall_at_k": {f"recall@{k}": recall_sums[k] / denom for k in range(1, args.max_k + 1)}
    }

    summary_path = args.save_path.replace(".jsonl", "_summary.json")
    with open(summary_path, "w", encoding="utf8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
