import os
import sys
from pathlib import Path
import re
import argparse
import json
import random
import unicodedata
import string
from typing import List, Optional
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.contriever import Contriever, M3contriver

DEFAULT_CHUNK_MODEL_PATH = "facebook/mcontriever-msmarco"


def resolve_chunk_model_path(contriever_path: str, chunk_model_path: Optional[str]) -> str:
    if chunk_model_path:
        return chunk_model_path
    if "mcontriever-msmarco" in contriever_path.lower():
        return contriever_path
    return DEFAULT_CHUNK_MODEL_PATH


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
def split_into_sentence(text: str) -> List[str]:
    sentences = re.split(r'(?<=[。！？?.!؟؛])\s*', text)
    return [s.strip() for s in sentences if s and s.strip()]
def normalize_text(text: str) -> str:
    text2 = text.translate(str.maketrans('', '', string.punctuation)).replace('  ', ' ').strip()
    if text2 and not text2.isspace():
        text = text2
    text2 = unicodedata.normalize('NFKD', text)
    if text2 and not text2.isspace():
        text = text2
    return text.lower().strip()
def build_chunks(document: str, tokenizer, lang: str, chunk_size: int = 200) -> List[str]:
    sentences = split_into_sentence(document)
    chunks, cur, cur_tok = [], "", 0
    joiner = "" if lang in ["zh", "ja", "ko"] else " "
    def flush():
        nonlocal cur, cur_tok
        if cur.strip():
            chunks.append(cur.strip())
        cur, cur_tok = "", 0
    for sent in sentences:
        toks = tokenizer.encode(sent, add_special_tokens=False)
        if len(toks) > chunk_size:                      # 单句超限：按 token 硬切
            flush()
            for i in range(0, len(toks), chunk_size):
                sub = tokenizer.decode(toks[i:i + chunk_size], skip_special_tokens=True).strip()
                if sub:
                    chunks.append(sub)
            continue
        if cur_tok + len(toks) <= chunk_size:
            cur += sent + joiner
            cur_tok += len(toks)
        else:
            flush()
            cur, cur_tok = sent + joiner, len(toks)
    flush()
    return chunks
def build_pool_for_lang(all_docs, lang, tokenizer, chunk_size=200):
    pool, seen = [], set()
    for doc in all_docs:
        if not doc:
            continue
        for chunk in build_chunks(doc, tokenizer, lang=lang, chunk_size=chunk_size):
            if chunk not in seen:
                seen.add(chunk)
                pool.append(chunk)
    return pool
@torch.no_grad()
def encode_texts(model, tokenizer, texts, device, batch_size=128, show_progress=False, desc="encode"):
    embs = []
    it = range(0, len(texts), batch_size)
    if show_progress:
        it = tqdm(it, desc=desc, leave=False)
    for i in it:
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        emb = model(**inputs)
        embs.append(emb.detach().cpu())
    return torch.cat(embs, dim=0) if embs else torch.empty(0)
def find_gold_chunk_ids(chunks, answers):
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
def recall_at_k(ranked_ids, gold_ids, k):
    if k <= 0 or not gold_ids:
        return 0.0
    topk = set(ranked_ids[:k])
    return 1.0 if any(gid in topk for gid in gold_ids) else 0.0
def ensure_list_answer(ans):
    if ans is None:
        return []
    if isinstance(ans, list):
        return [str(x) for x in ans]
    s = str(ans)
    if ", " in s:
        return [x.strip() for x in s.split(", ") if x.strip()]
    return [s.strip()] if s.strip() else []
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--language", type=str, default="en", choices=["en","zh","ja","fr","ar","ko","es","pt"])
    parser.add_argument("--doc_lang", type=str, default="en", choices=["en","zh","ja","fr","ar","ko","es","pt"])
    parser.add_argument("--contriever_path", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, default=200)
    parser.add_argument("--max_k", type=int, default=20)
    parser.add_argument("--fixed_k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--chunk_model_path", type=str, default=os.getenv("CHUNK_MODEL_PATH"))
    parser.add_argument("--save_path", type=str, default="./results/retriever_recall.jsonl")
    parser.add_argument("--skip_no_gold", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if args.fixed_k is not None:
        k_values = [args.fixed_k]
        print(f"Fixed k mode: only evaluating recall@{args.fixed_k}")
    else:
        k_values = list(range(1, args.max_k + 1))
        print(f"Range k mode: evaluating recall@1..{args.max_k}")
    contriever_name = ""
    contriever = None
    con_tok = None
    bm25 = None
    if "bm25" in args.contriever_path.lower():
        contriever_name = "bm25"
    elif "mcontriever-msmarco" in args.contriever_path.lower():
        contriever_name = "mcontriever-msmarco"
        contriever = Contriever.from_pretrained(args.contriever_path).to(device).eval()
        con_tok = AutoTokenizer.from_pretrained(args.contriever_path)
    else:
        contriever_name = "m3contriever"
        contriever = M3contriver(args.contriever_path)   # 自行管理设备
        con_tok = None
    chunk_model_path = resolve_chunk_model_path(args.contriever_path, args.chunk_model_path)
    chunk_tokenizer = AutoTokenizer.from_pretrained(chunk_model_path)
    chunk_tokenizer.model_max_length = 10**9   # 仅用于分块计数，屏蔽 >512 告警
    with open(args.dataset, 'r', encoding='utf8') as f:
        json_data = json.load(f)
        eval_data = json_data[args.language]
        eval_doc_lang = json_data[args.doc_lang]
    instances = []
    for instance, doc_lang_line in zip(eval_data, eval_doc_lang):
        instance["doc"] = doc_lang_line["doc"]
        instance["doc_lang_ans"] = doc_lang_line["QA"]["A"]
        instance["doc_lang"] = args.doc_lang
        instance["ori_lang_query"] = doc_lang_line["QA"]["Q"]
        if ', ' in instance["doc_lang_ans"]:
            instance["doc_lang_ans"] = instance["doc_lang_ans"].split(', ')
        instances.append(instance)
    all_docs = [ins["doc"] for ins in instances if ins.get("doc")]
    larger_pool = build_pool_for_lang(all_docs, args.doc_lang, chunk_tokenizer, chunk_size=args.chunk_size)
    print(f"Global pool built: {len(larger_pool)} chunks (doc_lang={args.doc_lang})")
    # ---- 关键提速：全局池只编码一次 ----
    pool_emb = None
    if contriever_name == "bm25":
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi([doc.split(" ") for doc in larger_pool])
    elif con_tok is not None:
        print("Encoding global pool once ...")
        pool_emb = encode_texts(contriever, con_tok, larger_pool, device,
                                batch_size=args.batch_size, show_progress=True, desc="pool")
    def rank(query: str):
        if con_tok is not None and contriever is not None:
            q_emb = encode_texts(contriever, con_tok, [query], device, args.batch_size)[0]
            scores = torch.matmul(pool_emb, q_emb)
        elif contriever is not None and con_tok is None:
            scores = torch.from_numpy(contriever(query, larger_pool))
        else:
            scores = torch.tensor(bm25.get_scores(query.split(" ")), dtype=torch.float)
        return torch.argsort(scores, descending=True).cpu().numpy().tolist()
    recall_sums = {k: 0.0 for k in k_values}
    used = 0
    skipped = 0
    max_k_for_topk = max(k_values)
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    with open(args.save_path, "w", encoding="utf8") as fw:
        for ins in tqdm(instances, desc=f"{args.language}->{args.doc_lang}"):
            query = ins["QA"]["Q"].replace("،", ",")
            gold_answers = ensure_list_answer(ins.get("doc_lang_ans"))
            doc_lang_ans_str = ins["doc_lang_ans"]
            if isinstance(doc_lang_ans_str, list):
                doc_lang_ans_str = " ".join(doc_lang_ans_str)
            else:
                doc_lang_ans_str = str(doc_lang_ans_str)
            retrieval_query = query + ' ' + doc_lang_ans_str
            gold_ids = find_gold_chunk_ids(larger_pool, gold_answers)
            if not gold_ids and args.skip_no_gold:
                skipped += 1
                continue
            ranked_ids = rank(retrieval_query)
            per_item = {
                "id": ins.get("id", None),
                "query": query,
                "ori_lang_query": ins["ori_lang_query"],
                "doc_lang": args.doc_lang,
                "contriever": contriever_name,
                "gold_answers": gold_answers,
                "gold_chunk_ids": gold_ids,
                "pool_size": len(larger_pool),
                "top_chunks": [larger_pool[i] for i in ranked_ids[:min(max_k_for_topk, len(larger_pool))]],
            }
            item_recalls = {}
            for k in k_values:
                r = recall_at_k(ranked_ids, gold_ids, k)
                recall_sums[k] += r
                item_recalls[f"recall@{k}"] = r
            per_item.update(item_recalls)
            fw.write(json.dumps(per_item, ensure_ascii=False) + "\n")
            used += 1
    denom = max(1, used)
    summary = {
        "dataset": args.dataset,
        "query_lang": args.language,
        "doc_lang": args.doc_lang,
        "contriever": contriever_name,
        "chunk_size_tokens": args.chunk_size,
        "fixed_k": args.fixed_k,
        "max_k": args.max_k if args.fixed_k is None else args.fixed_k,
        "seed": args.seed,
        "chunk_model_path": chunk_model_path,
        "pool_size": len(larger_pool),
        "used_instances": used,
        "skipped_no_gold": skipped,
        "recall_at_k": {f"recall@{k}": recall_sums[k] / denom for k in k_values},
    }
    summary_path = args.save_path.replace(".jsonl", "_summary.json")
    with open(summary_path, "w", encoding="utf8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("Done.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
if __name__ == "__main__":
    main()
