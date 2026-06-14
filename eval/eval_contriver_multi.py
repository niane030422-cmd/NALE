import os

import sys

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:

    sys.path.insert(0, str(REPO_ROOT))

import re

import json

import argparse

import random

import string

import unicodedata

from typing import List, Dict, Any, Tuple

from collections import defaultdict, Counter

import numpy as np

import torch

import pandas as pd

import matplotlib.pyplot as plt

from transformers import AutoTokenizer

from src.contriever import Contriever, M3contriver

from public_repo import repo_path

LANGS = ["en", "zh", "fr", "es", "pt", "ja", "ko", "ar"]

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

    text2 = text.translate(str.maketrans('', '', string.punctuation)).replace('  ', ' ').strip()

    if text2 and not text2.isspace():

        text = text2

    text2 = unicodedata.normalize('NFKD', text)

    if text2 and not text2.isspace():

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
    Supports:
      - contriever + tokenizer (HF encoder)
      - contriever callable (M3contriver)
      - BM25
    """

    if con_tok is not None and contriever is not None:

        sentences = [query] + documents

        inputs = con_tok(sentences, padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():

            embeddings = contriever(**inputs)

        q_emb = embeddings[0]

        d_emb = embeddings[1:]

        scores = torch.matmul(d_emb, q_emb)

        scores = scores.detach().cpu()

    elif contriever is not None and con_tok is None:

        scores = contriever(query, documents)

        scores = torch.from_numpy(scores)

    else:

        from rank_bm25 import BM25Okapi

        tokenized_corpus = [doc.split(" ") for doc in documents]

        bm25 = BM25Okapi(tokenized_corpus)

        tokenized_query = query.split(" ")

        scores = torch.tensor(bm25.get_scores(tokenized_query), dtype=torch.float)

    sorted_idx = torch.argsort(scores, descending=True).cpu().numpy().tolist()

    return sorted_idx

def hit_answer_type(

    chunk_text: str,

    chunk_lang: str,

    q_lang: str,

    ori_answer: Dict[str, str],

    new_answer: Dict[str, Dict[str, str]],

) -> str:

    """
    returns one of: "new", "ori", "both", "none"
    """

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

def load_instances(path: str) -> List[Dict[str, Any]]:

    """
    Supports two shapes:
      1) a single dict with keys: query, doc, ori_answer, new_answer
      2) a list of such dicts
    """

    with open(path, "r", encoding="utf8") as f:

        data = json.load(f)

    if isinstance(data, list):

        return data

    if isinstance(data, dict) and "query" in data and "doc" in data:

        return [data]

    raise ValueError("Unsupported dataset format. Expect dict with {query, doc, ori_answer, new_answer} or list of them.")

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True, help="json dataset path")

    parser.add_argument("--contriever_path", type=str, required=True, help="contriever path or bm25")

    parser.add_argument("--chunk_size", type=int, default=200, help="tokens per chunk")

    parser.add_argument("--top_k", type=int, default=15, help="topK chunks for analysis")

    parser.add_argument("--seed", type=int, default=233)

    parser.add_argument("--save_dir", type=str, default="./contriver_evalresults/multilang_analysis")

    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

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

    chunk_tokenizer = AutoTokenizer.from_pretrained(repo_path("models", "facebook", "mcontriever-msmarco"))

    instances = load_instances(args.dataset)

    lang_share_sum = {q: {d: 0.0 for d in LANGS} for q in LANGS}

    lang_share_cnt = {q: 0 for q in LANGS}

    answer_hit_sum = {q: Counter() for q in LANGS}

    per_query_records = []

    for inst_idx, ins in enumerate(instances):

        query_map: Dict[str, str] = ins["query"]

        doc_map: Dict[str, str] = ins["doc"]

        ori_answer: Dict[str, str] = ins.get("ori_answer", {})

        new_answer: Dict[str, Dict[str, str]] = ins.get("new_answer", {})

        for q_lang in LANGS:

            if q_lang not in query_map:

                continue

            query = str(query_map[q_lang]).replace("،", ",")

            pool_chunks: List[str] = []

            pool_langs: List[str] = []

            for d_lang in LANGS:

                doc = doc_map.get(d_lang, None)

                if not doc:

                    continue

                chunks = build_chunks(str(doc), chunk_tokenizer, lang=d_lang, chunk_size=args.chunk_size)

                pool_chunks.extend(chunks)

                pool_langs.extend([d_lang] * len(chunks))

            if not pool_chunks:

                continue

            ranked_ids = sort_documents_by_similarity(contriever, con_tok, query, pool_chunks)

            top_ids = ranked_ids[: min(args.top_k, len(ranked_ids))]

            top_langs = [pool_langs[i] for i in top_ids]

            lang_counts = Counter(top_langs)

            denom = max(1, len(top_ids))

            for d_lang in LANGS:

                lang_share_sum[q_lang][d_lang] += lang_counts.get(d_lang, 0) / denom

            lang_share_cnt[q_lang] += 1

            hit_counter = Counter()

            for i in top_ids:

                d_lang = pool_langs[i]

                ch = pool_chunks[i]

                t = hit_answer_type(ch, d_lang, q_lang, ori_answer, new_answer)

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

                    {"rank": r+1, "doc_lang": pool_langs[i], "text": pool_chunks[i][:220]}

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

    mat = df_heat_values.values

    plt.imshow(mat, aspect="auto", cmap="Blues")

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

        "top_k": args.top_k,

        "seed": args.seed,

        "avg_lang_share_n_per_q_lang": {q: int(lang_share_cnt[q]) for q in LANGS},

        "outputs": {

            "heatmap_csv": "topk_lang_share_heatmap.csv",

            "heatmap_png": "topk_lang_share_heatmap.png",

            "stackedbar_png": "topk_lang_share_stackedbar.png",

            "answer_hit_csv": "topk_answer_hit_types.csv",

            "answer_hit_png": "topk_answer_hit_types_stackedbar.png",

            "detail_jsonl": "per_query_detail.jsonl",

        }

    }

    with open(os.path.join(args.save_dir, "summary.json"), "w", encoding="utf8") as f:

        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Done.")

    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":

    main()
