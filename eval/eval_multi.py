import os

import re

import argparse

import json

import yaml

import sys

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:

    sys.path.insert(0, str(REPO_ROOT))

import tqdm

import random

import math

import torch

import numpy as np

from transformers import AutoModel, AutoTokenizer

import torch

import numpy as np

from src.contriever import Contriever, M3contriver

sys.path.append('..')

from llm.hf import Aya, Qwen

from llm.gpt import GPT

import re

import unicodedata

import string

def generate_3grams(str1):

    tmp_str = [str1[i:i+3] for i in range(len(str1) - 2)]

    str_list = [str1 for str1 in tmp_str if ' ' not in str1]

    return str_list

def compute_recall(list_a, str_b):

    if len(list_a) == 0:

        return 0.0

    match_count = 0

    for gram in list_a:

        if gram in str_b:

            match_count += 1

    recall_score = match_count / len(list_a)

    return recall_score

def calculate_3gram_recall(str_a, str_b):

    if len(str_a) < 3 or len(str_b) < 3:

        return 0.0

    str_a = str(str_a).replace('[\'', '').replace('\']', '').replace('[\"', '').replace('\"]', '').replace("', '", ' ').replace('", "', ' ')

    str_a_3grams = generate_3grams(str_a)

    recall_score = compute_recall(str_a_3grams, str_b)

    return recall_score

def normalize_text(text):

    debug_text = text

    original_text = text

    text = text.translate(str.maketrans('', '', string.punctuation)).replace('  ', ' ').strip()

    if text == '' or text.isspace() or not(text.isalpha() or text.isdigit()):

        text = original_text

    original_text = text

    text = unicodedata.normalize('NFKD', text)

    if text == '' or text.isspace() or not(text.isalpha() or text.isdigit()):

        text = original_text

    original_text = text

    text = text.lower().strip()

    return text

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

def sort_documents_by_similarity(contriever, tokenizer, sentence, documents, top_n, force_first_chunk=True):

    if tokenizer is not None :

        sentences = [sentence] + documents

        inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():

            embeddings = contriever(**inputs)

        sentence_embedding = embeddings[0]

        document_embeddings = embeddings[1:]

        similarity_scores = torch.matmul(document_embeddings, sentence_embedding)

    elif contriever is not None :

        similarity_scores = contriever(sentence, documents)

        similarity_scores = torch.from_numpy(similarity_scores)

    else:

        from rank_bm25 import BM25Okapi

        tokenized_corpus = [doc.split(" ") for doc in documents]

        bm25 = BM25Okapi(tokenized_corpus)

        tokenized_query = sentence.split(" ")

        similarity_scores = bm25.get_scores(tokenized_query)

        similarity_scores = torch.from_numpy(similarity_scores)

    sorted_indices = torch.argsort(similarity_scores, descending=True).numpy().tolist()

    if len(documents) < top_n:

        top_n = len(documents)

    top_n_ids = sorted_indices[:top_n]

    if force_first_chunk:

        if 0 in top_n_ids:

            top_n_ids.remove(0)

        if len(top_n_ids) == top_n:

            top_n_ids = [0] +  top_n_ids[:-1]

        else:

            top_n_ids = [0] +  top_n_ids

    sorted_documents = [documents[idx] for idx in top_n_ids]

    return sorted_documents

def split_into_sentence(text):

    sentence_endings = '[。！？?.!؟；;]'

    sentences = re.split(fr'(?<=({sentence_endings}))\s*', text)

    sentences = [sent.strip() for sent in sentences if sent]

    return sentences

def get_retrieval_query(query, label, retrieval_setting):

    if retrieval_setting == "given_gt":

        if isinstance(label, list):

            label = " ".join(map(str, label))

        else:

            label = str(label)

        return f"{query} {label}"

    return query

def split_into_chunks(document, label, chunk_num, mode, noise_rate, noise_doc, contriever, con_tok, query, lang, tokenizer, chunk_size=200, force_first_chunk=True, retrieval_setting="given_gt"):

    current_chunk = ""

    sentences = split_into_sentence(document)

    chunks = []

    current_token_num = 0

    for sent in sentences:

        tokens = tokenizer.encode(sent)

        if current_token_num + len(tokens) <= chunk_size:

            if lang in ["zh", "ja", "ko"]:

                current_chunk += sent

            else:

                current_chunk += sent + " "

            current_token_num += len(tokens)

        else:

            chunks.append(current_chunk.strip())

            current_chunk = sent

            current_token_num = len(tokens)

    if len(current_chunk) > 0 and current_chunk != ' ':

        chunks.append(current_chunk.strip())

    return_chunks = []

    ans_chunks = []

    index_list = []

    relevant_chunks = []

    retrieval_query = get_retrieval_query(query, label, retrieval_setting)

    return_chunks = sort_documents_by_similarity(contriever, con_tok, retrieval_query, chunks, top_n=chunk_num, force_first_chunk=force_first_chunk)

    return return_chunks

def processdata(instance, noise_rate, chunk_num, filename, language, mode="golden", noise_doc='', contriever=None, con_tok=None, tokenizer=None, languages=None, force_first_chunk=True, retrieval_setting="given_gt"):

    query = instance[f'query']

    ans = instance[f'new_answer']

    docs = instance['doc']

    ret_docs = []

    doc_lang = {}

    for lang in languages:

        retrieval_query = query[lang] if retrieval_setting == "given_gt" else query[language]

        tmp_docs = split_into_chunks(docs[lang], ans[lang][lang], chunk_num, mode, noise_rate, noise_doc, contriever, con_tok, retrieval_query, lang=lang, tokenizer=tokenizer, force_first_chunk=force_first_chunk, retrieval_setting=retrieval_setting)

        doc_lang[lang] = tmp_docs

    return query, ans, doc_lang

def split_by_stopwords(s, lst):

    pattern = '|'.join(map(re.escape, lst))

    res = re.split(pattern, s)

    res = [item.strip() for item in res if item != '']

    return res

def compare_texts(text1, text2, lang):

    normalized_text1 = normalize_text(text1)

    normalized_text2 = normalize_text(text2)

    ngram_res = calculate_3gram_recall(normalized_text1, normalized_text2)

    return ngram_res

def checkanswer(prediction, lang_gt, lang):

    prediction = prediction.lower()

    ngram_label = []

    for k, lang_gt in  lang_gt.items():

        tmp_ngram_label = []

        if type(lang_gt) is not list:

            lang_gt = [lang_gt]

        for instance in lang_gt:

            ngram_tmp = compare_texts(instance, prediction, lang=lang)

            tmp_ngram_label.append(ngram_tmp)

        if len(ngram_label) > 0:

            for i_lang in range(len(ngram_label)):

                ngram_label[i_lang] = max(ngram_label[i_lang], tmp_ngram_label[i_lang])

        else:

            ngram_label = tmp_ngram_label

    return ngram_label

def predict(query, ground_truth, lang_docs, model, system, instruction, dataset, language, languages=None):

    ret_doc = []

    docs = ''

    if len(lang_docs.keys()) == 0:

        text = query

        prediction = model.chat(prompt=text, temperature=0.8, max_tokens=256)

    else:

        for lang_d in lang_docs.values():

            lang_d = ' ... '.join(lang_d)

            ret_doc.append(lang_d)

        random.shuffle(ret_doc)

        random.shuffle(ret_doc)

        random.shuffle(ret_doc)

        docs = ''

        for doc in ret_doc:

            docs += f'{doc} '

        text = instruction.format(SYSTEM=system, QUERY=query, DOCS=docs)

        prediction = model.chat(prompt=text, temperature=0.8, max_tokens=256)

    if prediction is None:

        print("prediction is None")

        quit()

    elif type(prediction) is not str:

        if isinstance(prediction, dict):

            prediction = prediction["message"]["content"]

        else:

            prediction = prediction.content

    if '信息不足' in prediction or 'insufficient information' in prediction or '十分な情報がないため' in prediction or 'informations insuffisantes' in prediction or 'المستند يحتوي على معلومات غير كافية' in prediction or 'insuficiente en los documentos' in prediction or '문서 정보가 부족하여' in prediction or 'insuficientes no documento' in prediction or "información insuficiente" in prediction or "informações insuficientes" in prediction:

        lang_ngram = {}

    else:

        lang_ngram = {}

        for lang in languages:

            ngram_label = checkanswer(prediction, ground_truth[lang], lang)

            lang_ngram[lang] = ngram_label

    return lang_ngram, prediction, docs

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(

        '--modelname', type=str, default='bloomz',

        help='model name'

    )

    parser.add_argument(

        '--contriever_path', type=str,

        help='contriever_path'

    )

    parser.add_argument(

        '--dataset', type=str,

        help='evaluetion dataset'

    )

    parser.add_argument(

        '--language', type=str, default='en',

        help='query anguage',

        choices=["en", "zh", "fr", "ja", "ar", "es", "pt", "ko"]

    )

    parser.add_argument(

        '--doc_lang', type=str, default='all',

        help='document language',

        choices=["en", "zh", "fr", "ja", "ar", "es", "pt", "ko", 'all']

    )

    parser.add_argument(

        '--description', type=str, default='lang_specific',

        help='description mode',

        choices=['en','lang_specific']

    )

    parser.add_argument(

        '--mode', type=str, default='golden',

        help='document mode',

        choices=['golden', "conflict"]

    )

    parser.add_argument(

        '--noise_rate', type=float, default=0.0,

        help='rate of noisy chunks'

    )

    parser.add_argument(

        '--chunk_num', type=int, default=5,

        help='number of external chunks'

    )

    parser.add_argument(

        '--seed', type=int, default=233,

        help='random_seed'

    )

    parser.add_argument(

        '--language_mode', type=str, default="query_lang",

        help="noise or golden chunk language",

        choices=["en", "zh", "fr", "ja", "ar", "es", "pt", "ko", "query_lang"]

    )

    parser.add_argument(

        '--retrieval_setting', type=str, default='given_gt',

        choices=['given_gt', 'end2end'],

        help='given_gt uses per-language query+GT and forces chunk 0 into Top-K; end2end uses the selected query language only and the retriever Top-K directly'

    )

    args = parser.parse_args()

    languages = ["en", "zh", "fr", "ja", "ar", "es", "pt", "ko"]

    f1 = open(args.dataset)

    instances = json.load(f1)

    f1.close()

    noise_pool = {}

    modelname = args.modelname.split('/')[-1]

    args.dataset = args.dataset.split('/')[-1].replace('.json', '')

    retrieval_suffix = '' if args.retrieval_setting == 'given_gt' else '_end2end'

    resultpath = f'./results/multi_lingual/{args.dataset}_{args.description}/{args.mode}{retrieval_suffix}'

    os.makedirs(resultpath+f'/{modelname}', exist_ok=True)

    resultpath = resultpath+f'/{modelname}'

    prompt = yaml.load(open('./instructions/instruction_mul.yaml', 'r'), Loader=yaml.FullLoader)

    system = ''

    if args.description == 'lang_specific':

        system = prompt[args.language]['system']

    elif args.description == 'en':

        system = prompt['en_description']

    instruction = prompt[args.language]['instruction']

    if 'gpt' in args.modelname.lower():

        model = GPT(model=args.modelname)

    elif 'aya' in args.modelname.lower():

        model = Aya(model=args.modelname)

    elif 'qwen' in args.modelname.lower():

        model = Qwen(model=args.modelname)

    model_tokenizer = model.get_llm_tokenizer()

    contriever_name = ''

    if 'mcontriever-msmarco' in args.contriever_path.lower():

        contriever = Contriever.from_pretrained(args.contriever_path)

        con_tok = AutoTokenizer.from_pretrained(args.contriever_path)

        contriever_name = 'mcontriever-msmarco'

    else:

        if 'bm25' in args.contriever_path.lower():

            contriever = None

            con_tok = None

            contriever_name = 'bm25'

        else:

            contriever = M3contriver(args.contriever_path)

            con_tok = None

            contriever_name = 'm3contriever'

    os.makedirs(resultpath+f'/{contriever_name}', exist_ok=True)

    resultpath = resultpath+f'/{contriever_name}'

    filename = f'{resultpath}/prediction_n_{args.dataset}_ql{args.language}_multi_conflict_{modelname}_noise{args.noise_rate}_chunk{args.chunk_num}.json'

    results = []

    set_seed(int(args.seed))

    cnt = 0

    with open(filename,'w', encoding='utf8') as f:

        for instance in tqdm.tqdm(instances):

            if args.chunk_num == 0:

                query = instance[f'query']

                ans = instance[f'new_ans']

                docs = {}

            else:

                noise_doc = ''

                query, ans, lang_docs = processdata(instance, args.noise_rate, args.chunk_num, args.dataset, args.language, args.mode, noise_doc=noise_doc, contriever=contriever, con_tok=con_tok, tokenizer=model_tokenizer, languages=languages, force_first_chunk=(args.retrieval_setting == 'given_gt'), retrieval_setting=args.retrieval_setting)

            lang_ngram, prediction, docs = predict(query, ans, lang_docs, model, system, instruction, args.dataset, args.language, languages=languages)

            newinstance = {

                'id': cnt,

                f'query': query[args.language],

                f'ans': ans,

                f'prediction': prediction,

                f'docs': docs,

                f'noise_rate': args.noise_rate,

                f'retrieval_setting': args.retrieval_setting,

                f"lang_ngram": lang_ngram

            }

            results.append(newinstance)

            cnt += 1

            f.write(json.dumps(newinstance, ensure_ascii=False)+'\n')

    sum_ngram = {}

    ngram_num = 0

    for lang in languages:

        sum_ngram[lang] = 0

    for i in results:

        lang_ngram = i['lang_ngram']

        first_len = None

        for lang in languages:

            if lang in lang_ngram:

                sum_ngram[lang] += sum(lang_ngram[lang])

                if first_len is None:

                    first_len = len(lang_ngram[lang])

        if first_len is not None:

            ngram_num += first_len

    for lang in languages:

        if ngram_num > 0:

            sum_ngram[lang] = sum_ngram[lang] / ngram_num

        else :

            sum_ngram[lang] = None

    scores = {

    'language': args.language,

    'noise_rate': args.noise_rate,

    'retrieval_setting': args.retrieval_setting,

    'nums': len(results),

    'Character 3-gram Recall': sum_ngram

    }

    json.dump(scores,open(f'{resultpath}/prediction_{args.dataset}_ql{args.language}_lm{args.language_mode}_{args.modelname}_noise{args.noise_rate}_chunk{args.chunk_num}_seed{args.seed}_result.json','w'),ensure_ascii=False,indent=4)

    print ('Done')
