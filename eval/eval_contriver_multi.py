import os
import sys
from pathlib import Path
import re
import json
import argparse
import random
import string
import unicodedata
from typing import List, Dict, Any, Optional
from collections import Counter
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.contriever import Contriever, M3contriver
LANGS = ["en", "fr", "es", "pt", "zh", "ko", "ja", "ar"]
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
    sentences = re.split(r'(?<=[。！？?.!؟；;])\s*', text)
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
def rank_with_pool_emb(model, tokenizer, query, pool_emb, device, batch_size=128):
    q_emb = encode_texts(model, tokenizer, [query], device, batch_size)[0]
    scores = torch.matmul(pool_emb, q_emb)
    return torch.argsort(scores, descending=True).cpu().numpy().tolist()
def sort_documents_by_similarity(contriever, con_tok, query, documents, device="cpu", batch_size=128):
    if con_tok is not None and contriever is not None:
        q_emb = encode_texts(contriever, con_tok, [query], device, batch_size)[0]
        d_emb = encode_texts(contriever, con_tok, documents, device, batch_size)
        scores = torch.matmul(d_emb, q_emb)
    elif contriever is not None and con_tok is None:
        scores = torch.from_numpy(contriever(query, documents))
    else:
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi([doc.split(" ") for doc in documents])
        scores = torch.tensor(bm25.get_scores(query.split(" ")), dtype=torch.float)
    return torch.argsort(scores, descending=True).cpu().numpy().tolist()
def hit_answer_type(chunk_text, chunk_lang, q_lang, ori_answer, new_answer):
    ch = normalize_text(chunk_text)
    ori = normalize_text(str(ori_answer.get(chunk_lang, "") or ""))
    new = normalize_text(str(new_answer.get(q_lang, {}).get(chunk_lang, "") or ""))
    hit_ori = bool(ori) and (ori in ch)
    hit_new = bool(new) and (new in ch)
    if hit_ori and hit_new:
        return "both"
    if hit_new:
        return "new"
    if hit_ori:
        return "ori"
    return "none"
def load_instances(path):
    with open(path, "r", encoding="utf8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "query" in data and "doc" in data:
        return [data]
    raise ValueError("Unsupported dataset format. Expect dict with {query, doc, ori_answer, new_answer} or list of them.")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--contriever_path", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--chunk_model_path", type=str, default=os.getenv("CHUNK_MODEL_PATH"))
    parser.add_argument("--save_dir", type=str, default="./contriver_evalresults/multilang_analysis")
    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    contriever_name = ""
    contriever = None
    con_tok = None
    if "bm25" in args.contriever_path.lower():
        contriever_name = "bm25"
    elif "mcontriever-msmarco" in args.contriever_path.lower():
        contriever_name = "mcontriever-msmarco"
        contriever = Contriever.from_pretrained(args.contriever_path).to(device).eval()
        con_tok = AutoTokenizer.from_pretrained(args.contriever_path)
    else:
        contriever_name = "m3contriever"
        contriever = M3contriver(args.contriever_path)
        con_tok = None
    chunk_model_path = resolve_chunk_model_path(args.contriever_path, args.chunk_model_path)
    chunk_tokenizer = AutoTokenizer.from_pretrained(chunk_model_path)
    chunk_tokenizer.model_max_length = 10**9
    instances = load_instances(args.dataset)
    docs_by_lang = {lang: [] for lang in LANGS}
    for ins in instances:
        doc_map = ins.get("doc", {})
        for lang in LANGS:
            if lang not in doc_map or not doc_map[lang]:
                continue
            val = doc_map[lang]
            if isinstance(val, list):
                docs_by_lang[lang].extend([str(v) for v in val])
            else:
                docs_by_lang[lang].append(str(val))
    global_chunks_by_lang = {lang: [] for lang in LANGS}
    for d_lang in LANGS:
        if not docs_by_lang[d_lang]:
            continue
        all_docs_str = " ".join(docs_by_lang[d_lang])
        global_chunks_by_lang[d_lang] = build_chunks(all_docs_str, chunk_tokenizer, lang=d_lang, chunk_size=args.chunk_size)
    print(f"Global pool built from {len(instances)} instances")
    for d_lang in LANGS:
        if global_chunks_by_lang[d_lang]:
            print(f"  {d_lang}: {len(global_chunks_by_lang[d_lang])} chunks")
    # ---- 关键提速：每种语言的池只编码一次 ----
    pool_emb_by_lang = {}
    bm25_by_lang = {}
    if contriever_name == "bm25":
        from rank_bm25 import BM25Okapi
        for d_lang in LANGS:
            if global_chunks_by_lang[d_lang]:
                bm25_by_lang[d_lang] = BM25Okapi([doc.split(" ") for doc in global_chunks_by_lang[d_lang]])
    elif con_tok is not None:
        for d_lang in LANGS:
            if global_chunks_by_lang[d_lang]:
                pool_emb_by_lang[d_lang] = encode_texts(
                    contriever, con_tok, global_chunks_by_lang[d_lang],
                    device, args.batch_size, show_progress=True, desc=f"pool[{d_lang}]")
    def rank_pool(d_lang, query):
        if con_tok is not None and contriever is not None:
            return rank_with_pool_emb(contriever, con_tok, query, pool_emb_by_lang[d_lang], device, args.batch_size)
        elif contriever is not None and con_tok is None:
            return sort_documents_by_similarity(contriever, None, query, global_chunks_by_lang[d_lang], device, args.batch_size)
        else:
            scores = torch.tensor(bm25_by_lang[d_lang].get_scores(query.split(" ")), dtype=torch.float)
            return torch.argsort(scores, descending=True).cpu().numpy().tolist()
    lang_share_sum = {q: {d: 0.0 for d in LANGS} for q in LANGS}
    lang_share_cnt = {q: 0 for q in LANGS}
    answer_hit_sum = {q: Counter() for q in LANGS}
    per_query_records = []
    for inst_idx, ins in enumerate(tqdm(instances, desc="instances")):
        query_map = ins["query"]
        ori_answer = ins.get("ori_answer", {})
        new_answer = ins.get("new_answer", {})
        for q_lang in LANGS:
            if q_lang not in query_map:
                continue
            query = str(query_map[q_lang]).replace("،", ",")
            combined_chunks, combined_langs = [], []
            for d_lang in LANGS:
                if not global_chunks_by_lang[d_lang]:
                    continue
                d_query = str(query_map.get(d_lang, query)).replace("،", ",")
                d_chunks = global_chunks_by_lang[d_lang]
                ranked_ids_d = rank_pool(d_lang, d_query)
                top_ids_d = ranked_ids_d[: min(args.top_k, len(ranked_ids_d))]
                for i in top_ids_d:
                    combined_chunks.append(d_chunks[i])
                    combined_langs.append(d_lang)
            if not combined_chunks:
                continue
            ranked_ids = sort_documents_by_similarity(contriever, con_tok, query, combined_chunks, device, args.batch_size)
            top_ids = ranked_ids[: min(args.top_k, len(ranked_ids))]
            pool_chunks, pool_langs = combined_chunks, combined_langs
            top_langs = [pool_langs[i] for i in top_ids]
            lang_counts = Counter(top_langs)
            denom = max(1, len(top_ids))
            for d_lang in LANGS:
                lang_share_sum[q_lang][d_lang] += lang_counts.get(d_lang, 0) / denom
            lang_share_cnt[q_lang] += 1
            hit_counter = Counter()
            for i in top_ids:
                t = hit_answer_type(pool_chunks[i], pool_langs[i], q_lang, ori_answer, new_answer)
                hit_counter[t] += 1
            answer_hit_sum[q_lang].update(hit_counter)
            per_query_records.append({
                "instance_idx": inst_idx,
                "q_lang": q_lang,
                "retriever": contriever_name,
                "top_k": args.top_k,
                "lang_counts": dict(lang_counts),
                "answer_hit_counts": dict(hit_counter),
                "top_chunks_preview": [
                    {"rank": r + 1, "doc_lang": pool_langs[i], "text": pool_chunks[i][:220]}
                    for r, i in enumerate(top_ids[: min(5, len(top_ids))])
                ]
            })
    heat_data = []
    for q in LANGS:
        c = max(1, lang_share_cnt[q])
        row = {d: lang_share_sum[q][d] / c for d in LANGS}
        row["q_lang"] = q
        row["n_queries"] = lang_share_cnt[q]
        heat_data.append(row)
    df_heat = pd.DataFrame(heat_data).set_index("q_lang")
    df_heat_values = df_heat[LANGS].copy()
    df_heat.to_csv(os.path.join(args.save_dir, "topk_lang_share_heatmap.csv"), encoding="utf8")
    plt.figure()
    plt.imshow(df_heat_values.values, aspect="auto", cmap="Blues")
    plt.yticks(range(len(df_heat_values.index)), df_heat_values.index.tolist())
    plt.xticks(range(len(LANGS)), LANGS, rotation=45, ha="right")
    plt.colorbar()
    plt.title(f"Avg doc-language share in Top-{args.top_k} (retriever={contriever_name})")
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "topk_lang_share_heatmap.png"), dpi=200)
    plt.close()
    detail_path = os.path.join(args.save_dir, "per_query_detail.jsonl")
    with open(detail_path, "w", encoding="utf8") as fw:
        for r in per_query_records:
            fw.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "dataset": args.dataset,
        "retriever": contriever_name,
        "chunk_size_tokens": args.chunk_size,
        "chunk_model_path": chunk_model_path,
        "top_k": args.top_k,
        "seed": args.seed,
        "pool_size_per_lang": {lang: len(global_chunks_by_lang[lang]) for lang in LANGS},
        "avg_lang_share_n_per_q_lang": {q: int(lang_share_cnt[q]) for q in LANGS},
        "outputs": {
            "heatmap_csv": "topk_lang_share_heatmap.csv",
            "heatmap_png": "topk_lang_share_heatmap.png",
            "detail_jsonl": "per_query_detail.jsonl",
        }
    }
    with open(os.path.join(args.save_dir, "summary.json"), "w", encoding="utf8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("Done.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
if __name__ == "__main__":
    main()
