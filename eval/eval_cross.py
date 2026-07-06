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

def processdata(instance, noise_rate, chunk_num, filename, language, mode="golden", noise_doc='', contriever=None, con_tok=None,  tokenizer=None, force_first_chunk=True, retrieval_setting="given_gt"):

    query = instance["QA"][f'Q'].replace('،', ',')

    ret_ans = instance["QA"][f'A'].replace('،', ',')

    ori_lang = instance["doc_lang"]

    ori_lang_query = instance["ori_lang_query"]

    ori_ans = instance["doc_lang_ans"]

    if ', ' in ret_ans:

        ret_ans = ret_ans.split(', ')

    docs = instance['doc']

    retrieval_query = ori_lang_query if retrieval_setting == "given_gt" else query

    docs = split_into_chunks(docs, ori_ans, chunk_num, mode, noise_rate, noise_doc, contriever, con_tok, query=retrieval_query, lang=ori_lang, tokenizer=tokenizer, force_first_chunk=force_first_chunk, retrieval_setting=retrieval_setting)

    if not isinstance(docs, list):

        docs = [docs]

    return query, ret_ans, docs

def compare_texts(text1, text2, lang):

    normalized_text1 = normalize_text(text1)

    normalized_text2 = normalize_text(text2)

    ngram_res = calculate_3gram_recall(normalized_text1,normalized_text2)

    return ngram_res

def checkanswer(prediction, lang_gt, lang, answer_match_mode="loosen"):

    prediction = (prediction or "").lower()

    ngram_label = []

    code_switch_ngram = []

    def pad_to(lst, n, fill=0.0):

        if len(lst) < n:

            lst.extend([fill] * (n - len(lst)))

    for k, gt in lang_gt.items():

        if answer_match_mode == "strict" and k != lang:

            continue

        gt_list = gt if isinstance(gt, list) else [gt]

        tmp_ngram_label = []

        for instance in gt_list:

            tmp_ngram_label.append(compare_texts(instance, prediction, lang=lang))

        if k == lang:

            ngram_label = tmp_ngram_label

        if not code_switch_ngram:

            code_switch_ngram = tmp_ngram_label[:]

        else:

            n = max(len(code_switch_ngram), len(tmp_ngram_label))

            pad_to(code_switch_ngram, n, 0.0)

            pad_to(tmp_ngram_label, n, 0.0)

            for i in range(n):

                code_switch_ngram[i] = max(code_switch_ngram[i], tmp_ngram_label[i])

    return ngram_label, code_switch_ngram

def predict(query, ground_truth, docs, model, system, instruction, dataset, language, answer_match_mode="loosen"):

    if len(docs) == 0:

        text = query

        prediction = model.chat(prompt=text, temperature=0.8, max_tokens=256)

    else:

        docs = ' ... '.join(docs)

        text = instruction.format(SYSTEM=system, QUERY=query, DOCS=docs)

        prediction = model.chat(prompt=text, temperature=0.8, max_tokens=256)

    if prediction is None:

        print("prediction is None")

        quit()

    elif type(prediction) is not str:

        if hasattr(prediction, "content"):

            prediction = prediction.content

        elif isinstance(prediction, dict):

            prediction = prediction["message"]["content"]

        else:

            prediction = str(prediction)

    if '信息不足' in prediction or 'insufficient information' in prediction or '十分な情報がないため' in prediction or 'informations insuffisantes' in prediction or 'المستند يحتوي على معلومات غير كافية' in prediction or 'insuficiente en los documentos' in prediction or '문서 정보가 부족하여' in prediction or 'insuficientes no documento' in prediction or "información insuficiente" in prediction or "informações insuficientes" in prediction:

        ngram = [-1]

        code_switch_ngram = [-1]

    else:

        ngram, code_switch_ngram = checkanswer(prediction, ground_truth, language, answer_match_mode=answer_match_mode)

    return ngram, code_switch_ngram, prediction

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(

        '--modelname', type=str, default='bloomz',

        help='model name'

    )

    parser.add_argument(

        '--contriever_path', type=str,

        help='contriever path'

    )

    parser.add_argument(

        '--dataset', type=str,

        help='evaluetion dataset'

    )

    parser.add_argument(

        '--language', type=str, default='en',

        help='query anguage',

        choices=['en','zh','ja', 'fr', 'ar', 'ko', 'es', 'pt']

    )

    parser.add_argument(

        '--doc_lang', type=str, default='en',

        help='document language',

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

        choices=['golden','relevant','noise','hybird']

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

        '--retrieval_setting', type=str, default='given_gt',

        choices=['given_gt', 'end2end'],

        help='given_gt sets force_first_chunk=True and runs the if force_first_chunk block; end2end sets force_first_chunk=False and uses retriever Top-K directly'

    )

    parser.add_argument(

        '--answer_match_mode', type=str, default='loosen',

        choices=['loosen', 'strict'],

        help='loosen checks answers across languages; strict only checks the query language answer'

    )

    args = parser.parse_args()

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

    modelname = args.modelname.split('/')[-1]

    args.dataset = args.dataset.split('/')[-1].replace('json', '')

    retrieval_suffix = '' if args.retrieval_setting == 'given_gt' else '_end2end'

    resultpath = f'./results/cross_lingual/{args.dataset}_{args.description}/original_{args.answer_match_mode}{retrieval_suffix}'

    os.makedirs(resultpath+f'/{modelname}', exist_ok=True)

    resultpath = resultpath+f'/{modelname}'

    prompt = yaml.load(open('./instructions/instruction_cross_origin.yaml', 'r'), Loader=yaml.FullLoader)

    system = ''

    contriever_name = ''

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

    filename = f'{resultpath}/prediction_n_{args.dataset}_{contriever_name}_ql{args.language}_dl{args.doc_lang}_{modelname}_noise{args.noise_rate}_chunk{args.chunk_num}.json'

    results = []

    set_seed(int(args.seed))

    cnt = 0

    with open(filename,'w', encoding='utf8') as f:

        for instance in tqdm.tqdm(instances):

            tmp_3gram_score = 0

            acc_3gam_score = 0

            if args.chunk_num == 0:

                query = instance["QA"]['Q']

                ans = instance["QA"]['A'].replace('،', ',')

                if ', ' in ans:

                    ans = ans.split(', ')

                docs = []

            else:

                query, ans, docs = processdata(

                    instance=instance,

                    noise_rate=args.noise_rate,

                    chunk_num=args.chunk_num,

                    filename=filename,

                    language=args.language,

                    mode=args.mode,

                    noise_doc='',

                    contriever=contriever,

                    con_tok=con_tok,

                    tokenizer=model_tokenizer,

                    force_first_chunk=(args.retrieval_setting == 'given_gt'),

                    retrieval_setting=args.retrieval_setting

                )

            cnt += 1

            new_ans = {}

            new_ans[args.language] = ans

            new_ans[args.doc_lang] = instance["doc_lang_ans"]

            if ', ' in new_ans[args.doc_lang]:

                new_ans[args.doc_lang] = new_ans[args.doc_lang].split(', ')

            ans = new_ans

            ngram, code_switch_ngram, prediction = predict(query, ans, docs, model,system,instruction,args.dataset, args.language, answer_match_mode=args.answer_match_mode)

            newinstance = {

                'id': instance['id'],

                f'query': query,

                f'ans': ans,

                f'prediction': prediction,

                f'ngram': ngram,

                f'code_switch_ngram': code_switch_ngram,

                f'docs': docs,

                f'noise_rate': args.noise_rate,

                f'retrieval_setting': args.retrieval_setting,

                f'answer_match_mode': args.answer_match_mode

            }

            results.append(newinstance)

            f.write(json.dumps(newinstance, ensure_ascii=False)+'\n')

    sum_ngram = 0

    sum_code_switch_ngram = 0

    ngram_len = 0

    for i in results:

        sum_ngram += sum(i['ngram'])

        sum_code_switch_ngram += sum(i['code_switch_ngram'])

        ngram_len += len(i['ngram'])

    scores = {

    'character_3gram_recall_score': sum_ngram / ngram_len,

    'Code_Switch_Character 3-gram Recall': sum_code_switch_ngram / ngram_len,

    'noise_rate': args.noise_rate,

    'retrieval_setting': args.retrieval_setting,

    'answer_match_mode': args.answer_match_mode,

    'nums': len(results),

    }

    json.dump(scores,open(f'{resultpath}/prediction_{contriever_name}_{args.dataset}_ql{args.language}_dl{args.doc_lang}_{modelname}_noise{args.noise_rate}_chunk{args.chunk_num}_seed{args.seed}_result.json','w'),ensure_ascii=False,indent=4)

    print ('Done')
