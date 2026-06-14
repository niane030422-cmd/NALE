import os

import sys

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:

    sys.path.insert(0, str(REPO_ROOT))

import json

import glob

import argparse

from collections import defaultdict

from typing import Optional, Dict, Any

from public_repo import repo_path

LANGUAGES = ["en", "ko", "zh", "fr", "es", "pt", "ar", "ja"]

SEEDS = [233, 42, 31415]

def get_recall_dict(obj: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Read per-language 3-gram recall values from one result object."""

    for key in ["Character 3-gram Recall", "character 3-gram Recall"]:

        if key in obj and isinstance(obj[key], dict):

            out = {}

            for lang, v in obj[key].items():

                if lang in LANGUAGES:

                    out[lang] = float(v)

            return out if out else None

    return None

def load_one_file(fp: str):
    """Return query language and per-target-language recall data for one file."""

    try:

        with open(fp, "r", encoding="utf-8") as f:

            obj = json.load(f)

        q_lang = obj.get("language", None)

        recall_dict = get_recall_dict(obj)

        if q_lang not in LANGUAGES:

            q_lang = None

        return q_lang, recall_dict

    except Exception:

        return None, None

def mean(xs):

    return sum(xs) / len(xs) if xs else None

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--results_root", default=repo_path("results", "multi_lingual"))

    parser.add_argument("--out_dir", default=repo_path("avg_results", "multi"))

    parser.add_argument("--contriever", default="mcontriever-msmarco")

    args = parser.parse_args()

    model_list = ( 'gpt-5.1' ,  )

    contriever_name = args.contriever

    os.makedirs(args.out_dir, exist_ok=True)

    for model in model_list:

        scores = defaultdict(lambda: defaultdict(list))

        missing_files = []

        pattern = os.path.join(

            args.results_root,

            "**",

            model,

            contriever_name,

            "*_result.json"

        )

        matches = sorted(glob.glob(pattern, recursive=True))

        for fp in matches:

            try:

                with open(fp, "r", encoding="utf-8") as f:

                    obj = json.load(f)

                q_lang = obj.get("language", None)

                recall_dict = get_recall_dict(obj)

                if q_lang not in LANGUAGES or recall_dict is None:

                    missing_files.append(fp)

                    continue

                for t_lang in LANGUAGES:

                    if t_lang in recall_dict:

                        scores[q_lang][t_lang].append(float(recall_dict[t_lang]))

            except Exception:

                missing_files.append(fp)

        avg_matrix = {q: {} for q in LANGUAGES}

        detail = {q: {} for q in LANGUAGES}

        for q_lang in LANGUAGES:

            for t_lang in LANGUAGES:

                vals = scores[q_lang].get(t_lang, [])

                avg_matrix[q_lang][t_lang] = mean(vals)

                detail[q_lang][t_lang] = {

                    "mean": avg_matrix[q_lang][t_lang],

                    "count": len(vals),

                    "values": vals

                }

        out_path = os.path.join(args.out_dir, f"{contriever_name}_{model}_avg.json")

        out_obj = {

            "model": model,

            "contriever": contriever_name,

            "languages": LANGUAGES,

            "seeds": SEEDS,

            "avg_character_3gram_recall": avg_matrix,

            "detail": detail,

            "missing_files": missing_files

        }

        with open(out_path, "w", encoding="utf-8") as f:

            json.dump(out_obj, f, ensure_ascii=False, indent=2)

        print(f"[OK] Wrote: {out_path}")

        for q_lang in LANGUAGES:

            row_vals = [avg_matrix[q_lang][t] for t in LANGUAGES if avg_matrix[q_lang][t] is not None]

            row_mean = mean(row_vals)

            print(f"{q_lang}: row_mean={row_mean} (valid_pairs={len(row_vals)})")

if __name__ == "__main__":

    main()
