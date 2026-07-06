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

from src.contriever import Contriever,M3contriver

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

    if isinstance(label, list):

        label = " ".join(map(str, label))

    else:

        label = str(label)

    return f"{query} {label}"

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

def processdata(instance, noise_rate, chunk_num, filename, language, mode="golden", noise_doc='', contriever=None, con_tok=None, llm_tokenizer=None, force_first_chunk=True, retrieval_setting="given_gt"):

    query = instance["QA"][f'Q']

    ans = instance["QA"][f'A']

    if ', ' in ans:

        ans = ans.split(', ')

    docs = instance['doc']

    docs = split_into_chunks(docs, ans, chunk_num, mode, noise_rate, noise_doc, contriever, con_tok, query, lang=language,  tokenizer=llm_tokenizer, force_first_chunk=force_first_chunk, retrieval_setting=retrieval_setting)

    if not isinstance(docs, list):

        docs = [docs]

    return query, ans, docs

def compare_texts(text1, text2, lang):

    normalized_text1 = normalize_text(text1)

    normalized_text2 = normalize_text(text2)

    ngram_res = calculate_3gram_recall(normalized_text1,normalized_text2)

    return ngram_res

def checkanswer(prediction, ground_truth, lang):

    prediction = prediction.lower()

    if type(ground_truth) is not list:

        ground_truth = [ground_truth]

    ngram_labels = []

    for instance in ground_truth:

        ngram_res = compare_texts(instance, prediction, lang=lang)

        ngram_labels.append(ngram_res)

    return ngram_labels

def predict(query, ground_truth, docs, model, system, instruction, dataset, lang):

    if len(docs) == 0:

        text = query

        prediction = model.chat(prompt=text, temperature=0.8, max_tokens=256)

    else:

        docs = ' ... '.join(docs)

        text = instruction.format(SYSTEM=system, QUERY=query, DOCS=docs)

        prediction = model.chat(prompt=text,  temperature=0.8, max_tokens=256)

    if prediction is None: raise 'Wrong!'

    elif isinstance(prediction, dict):

        prediction = prediction.get("message", {}).get("content", "")

    else:

        prediction = getattr(prediction, "content", str(prediction))

    if '信息不足' in prediction or 'insufficient information' in prediction or '十分な情報がないため' in prediction or 'informations insuffisantes' in prediction or 'المستند يحتوي على معلومات غير كافية' in prediction or 'insuficiente en los documentos' in prediction or '문서 정보가 부족하여' in prediction or 'insuficientes no documento' in prediction or "información insuficiente" in prediction or "informações insuficientes" in prediction:

        ngram_labels = [-1]

    else:

        ngram_labels = checkanswer(prediction, ground_truth, lang)

    return ngram_labels, prediction

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(

        '--modelname', type=str, default='bloomz',

        help='model name'

    )

    parser.add_argument(

        '--contriever_path', type=str,

        help='model name'

    )

    parser.add_argument(

        '--dataset', type=str, default='50Q',

        help='evaluetion dataset',

    )

    parser.add_argument(

        '--language', type=str, default='en',

        help='language',

        choices=['en','zh','ja', 'fr', 'ar', 'ko', 'es', 'pt']

    )

    parser.add_argument(

        '--description', type=str, default='lang_specific',

        help='description mode',

        choices=['en','lang_specific']

    )

    parser.add_argument(

        '--mode', type=str, default='golden',

        help='chunk mode',

        choices=['golden']

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

        '--retrieval_setting', type=str, default='llm',

        choices=['llm', 'real', 'given_gt', 'end2end'],

        help='given_gt/llm sets force_first_chunk=True and runs the if force_first_chunk block; end2end/real sets force_first_chunk=False and uses retriever Top-K directly'

    )

    args = parser.parse_args()

    doc_id = 0

    instances = []

    noise_pool = {}

    with open(f'{args.dataset}','r', encoding='utf8') as f:

        eval_data = json.load(f)[args.language]

        for line in eval_data:

            instances.append(line)

            doc_id += 1

            noise_pool[doc_id] = line["doc"]

    modelname = args.modelname.split('/')[-1]

    args.dataset = args.dataset.split('/')[-1].replace('json', '')

    retrieval_setting = 'given_gt' if args.retrieval_setting == 'llm' else 'end2end' if args.retrieval_setting == 'real' else args.retrieval_setting

    retrieval_suffix = '' if retrieval_setting == 'given_gt' else '_end2end'

    resultpath = f'./results/mono/{args.dataset}_{args.description}/{args.mode}{retrieval_suffix}'

    os.makedirs(f'./results/mono/{args.dataset}_{args.description}/', exist_ok=True)

    if args.chunk_num == 0:

        resultpath = f'./results/worag/{args.dataset}_{args.description}'

        os.makedirs(f'./results/worag/', exist_ok=True)

    os.makedirs(resultpath, exist_ok=True)

    os.makedirs(resultpath+f'/new_{modelname}', exist_ok=True)

    resultpath = resultpath+f'/new_{modelname}'

    prompt = yaml.load(open('./instructions/instruction_mono.yaml', 'r'), Loader=yaml.FullLoader)

    system = ''

    if args.description == 'lang_specific':

        system = prompt[args.language]['system']

    elif args.description == 'en':

        system = prompt['en_description']

    instruction = prompt[args.language]['instruction']

    if 'qwen' in args.modelname.lower():

        model = Qwen(model=args.modelname)

    elif 'gpt' in args.modelname.lower():

        model = GPT(model=args.modelname)

    elif 'aya' in args.modelname.lower():

        model = Aya(model=args.modelname)

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

    if not os.path.exists(resultpath+f'/{contriever_name}'):

        os.mkdir(resultpath+f'/{contriever_name}')

    resultpath = resultpath+f'/{contriever_name}'

    filename = f'{resultpath}/prediction_n_{args.dataset}_{contriever_name}_{args.language}_{modelname}_noise{args.noise_rate}_chunk{args.chunk_num}.json'

    results = []

    cnt = 0

    with open(filename,'w', encoding='utf8') as f:

        for instance in tqdm.tqdm(instances):

            set_seed(int(args.seed))

            if args.chunk_num == 0:

                query = instance["QA"][f'Q']

                ans = instance["QA"][f'A']

                if ', ' in ans:

                    ans = ans.split(', ')

                docs = []

            else:

                noise_doc = ''

                while 1:

                    random_number = random.randint(0, doc_id-1)

                    if random_number != cnt:

                        noise_doc = noise_pool[random_number]

                        break

                query, ans, docs = processdata(instance, args.noise_rate, args.chunk_num, args.dataset, args.language, mode=args.mode, noise_doc=noise_doc, contriever=contriever, con_tok=con_tok, llm_tokenizer=model_tokenizer, force_first_chunk=(retrieval_setting == 'given_gt'), retrieval_setting=retrieval_setting)

            cnt += 1

            ngram_labels, prediction = predict(query, ans, docs, model, system, instruction, args.dataset, args.language)

            if not isinstance(ngram_labels,list):

                ngram_labels = [ngram_labels]

            newinstance = {

                'id': instance['id'],

                f'query': query,

                f'ans': ans,

                f'ngram_labels': ngram_labels,

                f'prediction': prediction,

                f'docs': docs,

                f'noise_rate': args.noise_rate

            }

            results.append(newinstance)

            f.write(json.dumps(newinstance, ensure_ascii=False)+'\n')

    n_gram_sum = 0

    result_num = 0

    for i in results:

        ngram_labels = i[f'ngram_labels']

        result_num += len(ngram_labels)

        n_gram_sum += sum(ngram_labels)

    scores = {

    'character 3-gram Recall': (n_gram_sum)/result_num if result_num > 0 else 0,

    'noise_rate': args.noise_rate,

    'retrieval_setting': retrieval_setting,

    'nums': len(results),

    }

    json.dump(scores,open(f'{resultpath}/prediction_{args.dataset}_{contriever_name}_{args.language}_{modelname}_noise{args.noise_rate}_chunk{args.chunk_num}_seed{args.seed}_result.json','w'),ensure_ascii=False,indent=4)

    print ('Done')
