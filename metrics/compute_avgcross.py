import os

import sys

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:

    sys.path.insert(0, str(REPO_ROOT))

import json

import glob

import argparse

import numpy as np

from collections import defaultdict

from public_repo import repo_path

LANGUAGES = ["en", "ko", "zh", "fr", "es", "pt", "ar", "ja"]

SEEDS = [233, 42, 31415]

def load_score(fp: str) -> float | None:

    try:

        with open(fp, "r", encoding="utf-8") as f:

            obj = json.load(f)

        if "Code_Switch_Character 3-gram Recall" in obj:

            return float(obj["Code_Switch_Character 3-gram Recall"])

    except Exception:

        return None

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--results_root", default=repo_path("results", "cross_lingual", "MRAG_dataset._lang_specific", "original_loosen"), help="mono results root")

    parser.add_argument("--out_dir", default=repo_path("avg_results", "cross", "original_loosen"), help="output dir")

    args = parser.parse_args()

    ans = []

    model_list=('Qwen3-30B-A3B-Instruct-2507',)

    for model in model_list:

        contriever_name = "mcontriever-msmarco"

        scores = defaultdict(lambda: defaultdict(list))

        files_used = defaultdict(lambda: defaultdict(list))

        missing = []

        for q_lang in LANGUAGES:

            for doc_lang in LANGUAGES:

                pattern = os.path.join(

                    args.results_root,

                    model,

                    contriever_name,

                    f"*_ql{q_lang}_dl{doc_lang}_{model}_*_result.json"

                )

                matches = glob.glob(pattern, recursive=True)

                for fp in matches:

                    score = load_score(fp)

                    if score is not None:

                        scores[q_lang][doc_lang].append(score)

                        files_used[q_lang][doc_lang].append(fp)

                    else:

                        missing.append(fp)

        avg_by_pair  = {}

        detail = {}

        for q_lang in LANGUAGES:

            for doc_lang in LANGUAGES:

                key = f"{q_lang}->{doc_lang}"

                vals = scores[q_lang].get(doc_lang, [])

                for values in vals:

                    ans.append(values)

                if vals:

                    mean = sum(vals) / len(vals)

                    avg_by_pair[key] = mean

                    detail[key] = {"mean": mean, "count": len(vals)}

                else:

                    avg_by_pair[key] = None

                    detail[key] = {"mean": None, "count": 0}

        os.makedirs(args.out_dir, exist_ok=True)

        out_path = os.path.join(args.out_dir, f"{contriever_name}_{model}.json")

        out_obj = {

            "model": model,

            "contriever": contriever_name,

            "seeds": SEEDS,

            "languages": LANGUAGES,

            "avg_character_3gram_recall": avg_by_pair,

            "detail": detail,

            "missing_files": missing,

        }

        with open(out_path, "w", encoding="utf-8") as f:

            json.dump(out_obj, f, ensure_ascii=False, indent=2)

        print(f"[OK] Wrote: {out_path}")

        for q_lang in LANGUAGES:

            vals = []

            for doc_lang in LANGUAGES:

                pair_key = f"{q_lang}->{doc_lang}"

                v = avg_by_pair[pair_key]

                if v is not None:

                    vals.append(v)

            if vals:

                print(f"{q_lang}: mean_over_doc_langs={sum(vals)/len(vals):.6f} (pairs={len(vals)})")

            else:

                print(f"{q_lang}: no data")

        ans=np.array(ans, dtype=float)

        print(f"Overall mean across all pairs: {ans.mean():.6f} (q_langs={len(ans)})")

        print(f"variance across q_langs: {ans.var():.6f}")

if __name__ == "__main__":

    main()
