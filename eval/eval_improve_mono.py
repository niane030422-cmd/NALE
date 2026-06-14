import os

import sys

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:

    sys.path.insert(0, str(REPO_ROOT))

import argparse

import json

import yaml

import tqdm

import random

import torch

import numpy as np

from transformers import AutoTokenizer

from llm.hf import Aya, Qwen

import re

import unicodedata

import string

def generate_3grams(str1):

    tmp_str = [str1[i:i+3] for i in range(len(str1) - 2)]

    str_list = [gram for gram in tmp_str if ' ' not in gram]

    return str_list

def compute_recall(list_a, str_b):

    if len(list_a) == 0:

        return 0.0

    match_count = 0

    for gram in list_a:

        if gram in str_b:

            match_count += 1

    return match_count / len(list_a)

def calculate_3gram_recall(str_a, str_b):

    if len(str_a) < 3 or len(str_b) < 3:

        return 0.0

    str_a = str(str_a).replace('[\'', '').replace('\']', '').replace('[\"', '').replace('\"]', '').replace("', '", ' ').replace('", "', ' ')

    str_a_3grams = generate_3grams(str_a)

    return compute_recall(str_a_3grams, str_b)

def normalize_text(text):

    if not isinstance(text, str):

        text = str(text)

    original_text = text

    text = text.translate(str.maketrans('', '', string.punctuation)).replace('  ', ' ').strip()

    if text == '' or text.isspace():

        text = original_text

    text = unicodedata.normalize('NFKD', text)

    return text.lower().strip()

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

def processdata(instance):
    """Extract query, answer, and pre-chunked docs from one instance."""

    query = instance.get("query", "")

    ans = instance.get("ans", "")

    if isinstance(ans, str) and ', ' in ans:

        ans = ans.split(', ')

    docs = instance.get('docs', [])

    if not isinstance(docs, list):

        docs = [str(docs)]

    else:

        docs = [str(d) for d in docs]

    return query, ans, docs

def compare_texts(text1, text2):

    normalized_text1 = normalize_text(text1)

    normalized_text2 = normalize_text(text2)

    return calculate_3gram_recall(normalized_text1, normalized_text2)

def checkanswer(prediction, ground_truth):

    prediction = prediction.lower()

    if not isinstance(ground_truth, list):

        ground_truth = [ground_truth]

    ngram_labels = []

    for gt in ground_truth:

        ngram_res = compare_texts(gt, prediction)

        ngram_labels.append(ngram_res)

    return ngram_labels

def predict(query, ground_truth, docs, model, system, instruction, lang):

    if not docs:

        text = query

        prediction = model.chat(prompt=text, temperature=0.8, max_tokens=256)

    else:

        context = ' ... '.join(docs)

        text = instruction.format(SYSTEM=system, QUERY=query, DOCS=context)

        prediction = model.chat(prompt=text, temperature=0.8, max_tokens=256)

    if prediction is None:

        return [0.0], "ERROR"

    if isinstance(prediction, dict):

        prediction = prediction.get("message", {}).get("content", "")

    else:

        prediction = getattr(prediction, "content", str(prediction))

    refuse_keywords = [

        '信息不足', 'insufficient information', '十分な情報がないため',

        'informations insuffisantes', 'المستند يحتوي على معلومات غير كافية',

        'insuficiente en los documentos', '문서 정보가 부족하여',

        'insuficientes no documento', "información insuficiente", "informações insuficientes"

    ]

    if any(kw in prediction for kw in refuse_keywords):

        ngram_labels = [-1]

    else:

        ngram_labels = checkanswer(prediction, ground_truth)

    return ngram_labels, prediction

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--modelname', type=str, default='Qwen3-8B')

    parser.add_argument('--dataset', type=str, required=True, help='Path to your JSON file with chunks')

    parser.add_argument('--language', type=str, default='en', choices=['en', 'zh'])

    parser.add_argument('--description', type=str, default='lang_specific', choices=['en', 'lang_specific'])

    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--retriever_name', type=str, default='pre_chunked')

    args = parser.parse_args()

    set_seed(args.seed)

    with open(args.dataset, 'r', encoding='utf8') as f:

        instances = json.load(f)

    model_short_name = args.modelname.split('/')[-1]

    dataset_name = os.path.basename(args.dataset).replace('.json', '')

    output_dir = f'./results/mono/{dataset_name}/{model_short_name}2/{args.retriever_name}'

    os.makedirs(output_dir, exist_ok=True)

    filename = f'{output_dir}/prediction_{args.language}_seed{args.seed}.json'

    prompt_config = yaml.load(open('./instructions/instruction_mono.yaml', 'r'), Loader=yaml.FullLoader)

    system = prompt_config[args.language]['system'] if args.description == 'lang_specific' else prompt_config['en_description']

    instruction = prompt_config[args.language]['instruction']

    if 'qwen' in args.modelname.lower():

        model = Qwen(model=args.modelname)

    elif 'aya' in args.modelname.lower():

        model = Aya(model=args.modelname)

    results = []

    with open(filename, 'w', encoding='utf8') as f:

        for instance in tqdm.tqdm(instances):

            query, ans, docs = processdata(instance)

            ngram_labels, prediction = predict(query, ans, docs, model, system, instruction, args.language)

            new_instance = {

                'id': instance.get('id', 'N/A'),

                'query': query,

                'ans': ans,

                'ngram_labels': ngram_labels,

                'prediction': prediction,

                'docs': docs

            }

            results.append(new_instance)

            f.write(json.dumps(new_instance, ensure_ascii=False) + '\n')

    n_gram_sum = 0

    valid_count = 0

    refuse_count = 0

    for res in results:

        labels = res['ngram_labels']

        if labels == [-1]:

            refuse_count += 1

            n_gram_sum += 0

            valid_count += 1

        else:

            n_gram_sum += sum(labels) / len(labels)

            valid_count += 1

    final_scores = {

        'avg_character_3gram_recall': n_gram_sum / valid_count if valid_count > 0 else 0,

        'refuse_rate': refuse_count / len(results) if len(results) > 0 else 0,

        'total_nums': len(results)

    }

    result_file = filename.replace('.json', '_result.json')

    with open(result_file, 'w', encoding='utf8') as f:

        json.dump(final_scores, f, ensure_ascii=False, indent=4)

    print(f"Done! Average Recall: {final_scores['avg_character_3gram_recall']:.4f}")
