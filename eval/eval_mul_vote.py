
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
from src.contriever import Contriever
import langid
from copy import deepcopy
sys.path.append('..')
#from llm.gpt_dashscope import GPT_4OL
#from llm.hf import Mixtral, LLaMA, Aya, LLaMA3, Baichuan, GLM, GEMMA
#from llm.gpt import GPT
#from llm.qwen_dashscope import QwenChatAtDS
from llm.hf import Qwen
import re
import unicodedata
import string

import stanza

from public_repo import repo_path

def split_last_space(s):
    # 找到最后一个空格的位置
    last_space_index = s.rfind(' ')
    
    # 如果没有找到空格，返回原始字符串和空字符串
    if last_space_index == -1:
        return s, ''
    
    # 以最后一个空格的位置将字符串切分为两部分
    part1 = s[:last_space_index]
    part2 = s[last_space_index + 1:]
    
    return part1, part2

def split_last_tk(s, tk):
    # 找到最后一个空格的位置
    last_space_index = s.rfind(tk)
    
    # 如果没有找到空格，返回原始字符串和空字符串
    if last_space_index == -1:
        return s, ''
    
    # 以最后一个空格的位置将字符串切分为两部分
    part1 = s[:last_space_index].strip()
    part2 = s[last_space_index + 1:].strip()
    
    return part1, part2

def doc_replace_ans(replace_note, ori_ans, new_ans, doc, lang):
    if lang not in ["zh", 'ja']:
        ori_ans_list = split_last_space(ori_ans)
        new_ans_list = split_last_space(new_ans)
        for i1, i2 in zip(ori_ans_list, new_ans_list):
            doc = doc.replace(i1, i2)
    else:
        if lang == 'zh' and "zh" not in replace_note.keys():
            doc = doc.replace(ori_ans, new_ans)
        else:
            ori_ans_list = split_last_tk(ori_ans, replace_note[lang])
            new_ans_list = split_last_tk(new_ans, replace_note[lang])
            for i1, i2 in zip(ori_ans_list, new_ans_list):
                doc = doc.replace(i1, i2)
    return doc

def detect_language(text):
    target_languages = {'zh', 'en', 'fr', 'ja', 'ko', 'es', 'pt', 'ar'}
    ranked_langs = langid.rank(text)
    filtered_probs = [(lang, prob) for lang, prob in ranked_langs if lang in target_languages]
    if filtered_probs:
        highest_prob_lang = max(filtered_probs, key=lambda x: x[1])[0]
    else:
        highest_prob_lang = "unknown"
    return highest_prob_lang


def lemmatize_text(text, language, stanza_pipeline):
    """
    对输入文本进行词形还原
    """
    # nlp = stanza.Pipeline(language, dir=model_dir, download_method=None, processors='tokenize,pos,lemma')
    # return ' '.join(lemmatized_words)
    try:
        doc = stanza_pipeline(text)
        lemmatized_words = [word.lemma for sent in doc.sentences for word in sent.words]
        if language not in ["zh", "ja"]:
            return ' '.join(lemmatized_words)
        else:
            return ''.join(lemmatized_words)
    except Exception as e:
        print(e)
        return text

def normalize_text(text, lang, stanza_pipeline):
    debug_text = text
    original_text = text
    
    # 移除标点符号
    text = text.translate(str.maketrans('', '', string.punctuation)).replace('  ', ' ').strip()
    # text = text.replace
    if text == '' or text.isspace() or not(text.isalpha() or text.isdigit()):
        text = original_text
    original_text = text
    
    # 标准化为 NFKD 形式并转换为小写
    text = unicodedata.normalize('NFKD', text)
    if text == '' or text.isspace() or not(text.isalpha() or text.isdigit()):
        text = original_text
    original_text = text
    text = text.lower().strip()

    return text

def split_by_stopwords(s, lst):
    pattern = '|'.join(map(re.escape, lst))
    res = re.split(pattern, s)
    res = [item.strip() for item in res if item != '']
    return res

def compare_texts(text1, text2, stop_list, lang, stanza_pipeline):
    text1_lang = detect_language(text1)
    text2_lang = detect_language(text2)
    normalized_text1 = normalize_text(text1, text1_lang, stanza_pipeline=None)
    normalized_text2 = normalize_text(text2, text2_lang, stanza_pipeline=None)
    
    # stop_word enhanced
    stop_word_flag = True
    ans_list = split_by_stopwords(normalized_text1, stop_list[text1_lang])    
    if not isinstance(ans_list, list):
        ans_list = [ans_list]
    stop_word_flag = True

    origianl_ans_list = split_by_stopwords(text1, stop_list[text1_lang])    
    if not isinstance(origianl_ans_list, list):
        origianl_ans_list = [origianl_ans_list]

    for itm, o_itm in zip(ans_list, origianl_ans_list):
        if itm.lower() not in normalized_text2.lower() and itm.lower() not in text2.lower() and o_itm.lower() not in normalized_text2.lower() and o_itm.lower() not in text2.lower():
            stop_word_flag = False
    
    # EM
    em_flag = True
    if text1.lower() not in text2.lower():
        em_flag = False

    return stop_word_flag, em_flag

def merge_lists(list1, list2):
    if isinstance(list1, list) and isinstance(list2, list):
        merged_list = []
        for item1, item2 in zip(list1, list2):
            if isinstance(item1, list) and isinstance(item2, list):
                merged_list.append(merge_lists(item1, item2))
            elif item1 == 1 or item2 == 1:
                merged_list.append(1)
            else:
                merged_list.append(0)
        return merged_list
    else:
        return None

def checkanswer(prediction, lang_gt, lang, stanza_pipeline=None, languages=None):
    # sw_labels, em_labels, ng_all
    prediction = prediction.lower()
    ret_sw_labels = []
    ret_em_labels = []

    # if ', ' in prediction:
    #     prediction = prediction.split(', ')

    stop_list = {}
    sw_labels = {}
    em_labels = {}
    ng_all = {}
    ng_all['max_lang'] = ''
    ng_all['max'] = 0
    f1 = open('./data/stopwords-all.json')
    json_df = json.load(f1)
    f1.close()
    for lang in languages:
        sw_labels[lang] = []
        em_labels[lang] = []
        ng_all[lang] = 0
        tmp_list = json_df[lang]
        if lang in ["en", "fr", "ar", "es", "pt", 'ko']:
            tmp_list = [f' {item} ' for item in tmp_list ]
        stop_list[lang] = tmp_list

    for lang in languages:
        true_lang_gt = lang_gt[lang]
        if ', ' in str(true_lang_gt) and not isinstance(true_lang_gt, list):
            true_lang_gt = true_lang_gt.split(', ')
        if type(true_lang_gt) is not list:
            true_lang_gt = [true_lang_gt]
        for instance in true_lang_gt:
            stop_flag = True
            em_flag = True
            
            tstop_flag, tem_flag = compare_texts(instance, str(prediction), stop_list, lang=lang, stanza_pipeline=stanza_pipeline)
            if not tstop_flag:
                stop_flag = False
            if not tem_flag:
                em_flag = False 
                    
            sw_labels[lang].append(int(stop_flag))
            em_labels[lang].append(int(em_flag))
        ng_all[lang] = calculate_3gram_recall(str(lang_gt[lang]), str(prediction))
    
    res_merge_sw = []
    res_merge_em = []
    for lang in languages:
        ng_all['max'] = max(ng_all['max'], ng_all[lang])
        if ng_all['max'] == ng_all[lang] and ng_all[lang] > 0:
            ng_all['max_lang'] = lang
        if len(res_merge_sw) == 0:
            res_merge_sw = sw_labels[lang]
            res_merge_em = em_labels[lang]
        else:
            res_merge_sw = merge_lists(sw_labels[lang], res_merge_sw)
            res_merge_em = merge_lists(em_labels[lang], res_merge_em)
    sw_labels['merge'] = res_merge_sw
    em_labels['merge'] = res_merge_em

    return sw_labels, em_labels, ng_all

def generate_3grams(s):
    tmp_str = [s[i:i+3] for i in range(len(s) - 2)]
    str_list = [s for s in tmp_str if ' ' not in s]
    return str_list

def compute_recall(list_a, list_b):
    if len(list_a) == 0:
        return 0.0

    match_count = 0
    b_count = {}
    
    # 计数 b 中的3-gram
    # print(list_b)
    for gram in list_b:

        if gram in b_count:
            b_count[gram] += 1
        else:
            b_count[gram] = 1

    # 计算匹配的3-gram数目
    for gram in list_a:
        # print(gram)
        # print(b_count)
        if gram in b_count and b_count[gram] > 0:
            match_count += 1
            b_count[gram] -= 1

    recall_score = match_count / len(list_a)
    # print(list_a, list_b, recall_score)
    return recall_score

def calculate_3gram_recall(a, b):
    # print(a,b)
    if len(a) < 3 or len(b) < 3:
        return 0.0  # 在字符串长度不足3时，返回0.0
    
    a = str(a).replace('[\'', '').replace('\']', '').replace('[\"', '').replace('\"]', '').replace("', '", ' ').replace('", "', ' ')

    a_3grams = generate_3grams(a)
    b_3grams = generate_3grams(b)

    recall_score = compute_recall(a_3grams, b_3grams)
    return recall_score

def ensure_dir_exists(dir_path):
    """
    检查指定的目录路径是否存在，如果不存在，则创建该目录。
    
    :param dir_path: 要检查或创建的目录路径
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"目录 '{dir_path}' 已创建。")
    else:
        print(f"目录 '{dir_path}' 已存在。")

def get_all_files_in_dir(dir_path):
    """
    获取指定目录路径下的所有文件列表。
    
    :param dir_path: 要查询的目录路径
    :return: 包含该目录下所有文件名的列表
    """
    files = []
    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        if os.path.isfile(item_path):
            files.append(item_path)
    return files

def res_process(prediction, ans_dict, stanza_list, languages):
    # print(f'res_process:{languages}')
    ngram_all = {}
    for lang in languages:
        ngram_all[lang] = {}
        ngram_all[lang]['max'] = 0
        for lang2 in languages:
            ngram_all[lang][lang2] = 0

    if '信息不足' in prediction or 'insufficient information' in prediction or '十分な情報がないため' in prediction or 'informations insuffisantes' in prediction or 'المستند يحتوي على معلومات غير كافية' in prediction or 'insuficiente en los documentos' in prediction or '문서 정보가 부족하여' in prediction or 'insuficientes no documento' in prediction or "información insuficiente" in prediction or "informações insuficientes" in prediction:
        accept_sw_labels = {"NONE":-1}
        accept_em_labels = {"NONE":-1}

    else:
        accept_sw_labels = {}
        accept_em_labels = {}
        for lang in languages:
            accept_sw_labels[lang] = {}
            accept_em_labels[lang] = {}
            accept_sw_labels[lang]['merge'] = []
            accept_em_labels[lang]['merge'] = []
            accept_sw_labels[lang]['label'] = 0
            accept_em_labels[lang]['label'] = 0
            for lang2 in languages:
                accept_sw_labels[lang][lang2] = {}
                accept_em_labels[lang][lang2] = {}

        for lang in languages:
            sw_labels, em_labels, ng_all = checkanswer(prediction, ans_dict[lang], lang, stanza_pipeline=stanza_list, languages=languages)
            accept_sw_labels[lang] = sw_labels
            accept_em_labels[lang] = em_labels
            ngram_all[lang] = ng_all

        if 'sampled_languages' in ans_dict.keys():
            sampled_language = ans_dict['sampled_languages'][0]
            
            accept_sw_labels['sampled_languages'] = accept_sw_labels[sampled_language]
            accept_em_labels['sampled_languages'] = accept_em_labels[sampled_language]
            ngram_all['sampled_languages'] = ng_all[sampled_language]
            

    factlabel = 0

    if '事实性错误' in prediction or 'factual errors' in prediction or '事実誤認' in prediction or "هناك أخطا" in prediction or 'erreurs factuelles' in prediction or "errores factuales" in prediction or 'erros factuais' in prediction or '사실적 오류가 있습니다' in prediction:
        factlabel = 1
    

    
    return accept_sw_labels, accept_em_labels, ngram_all, prediction, factlabel

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def sort_documents_by_similarity(contriever, tokenizer, sentence, documents, top_n):
    sentences = [sentence] + documents

    # 使用 Tokenizer 对输入进行编码
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

    # 使用 Contriever 模型计算嵌入向量
    with torch.no_grad():
        embeddings = contriever(**inputs)  # 获取 CLS token 的嵌入向量

    # 计算句子（第一个嵌入向量）与每个文档（后续嵌入向量）的内积相似度
    sentence_embedding = embeddings[0]
    document_embeddings = embeddings[1:]
    similarity_scores = torch.matmul(document_embeddings, sentence_embedding)

    # print(similarity_scores)

    # 获取相似度的排序索引
    sorted_indices = torch.argsort(similarity_scores, descending=True).numpy().tolist()

    if len(documents) < top_n:
        top_n = len(documents)
    
    # 确保第一段一定出现
    top_n_ids = sorted_indices[:top_n]

    if 0 in top_n_ids:
        top_n_ids.remove(0)

    # if 0 not in top_n_ids:
    if len(top_n_ids) == top_n:
       top_n_ids = [0] +  top_n_ids[:-1]
    else:
        top_n_ids = [0] +  top_n_ids

    # top_n_ids = sorted(top_n_ids, reverse=False)

    sorted_documents = [documents[idx] for idx in top_n_ids]

    # 返回按相似度排序后的文档
    # sorted_documents = [documents[idx] for idx in sorted_indices]
    # sorted_documents= []
    return sorted_documents

def split_into_sentence(text):
    sentence_endings = '[。！？?.!؟；;]'
    sentences = re.split(fr'(?<=({sentence_endings}))\s*', text)
    sentences = [sent.strip() for sent in sentences if sent]
    return sentences

def split_into_chunks(document, label, chunk_num, mode, noise_rate, noise_doc, contriever, con_tok, query, lang, stanza_pipeline, tokenizer, chunk_size=200):

    original_label = label

    # document
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
    
    similar_chunks = sort_documents_by_similarity(contriever, con_tok, str(query)+' '+str(original_label), chunks, top_n=chunk_num)

    return similar_chunks

def split_last_space(s):
    # 找到最后一个空格的位置
    last_space_index = s.rfind(' ')
    
    # 如果没有找到空格，返回原始字符串和空字符串
    if last_space_index == -1:
        return s, ''
    
    # 以最后一个空格的位置将字符串切分为两部分
    part1 = s[:last_space_index]
    part2 = s[last_space_index + 1:]
    
    return part1, part2

def split_last_tk(s, tk):
    if tk not in s:
        return s
    else:
        # 找到最后一个空格的位置
        last_space_index = s.rfind(tk)
        
        # 如果没有找到空格，返回原始字符串和空字符串
        if last_space_index == -1:
            return s, ''
        
        # 以最后一个空格的位置将字符串切分为两部分
        part1 = s[:last_space_index].strip()
        part2 = s[last_space_index + 1:].strip()
        
        return part1, part2

def doc_replace_entity(ori_ans_list, new_ans_list, doc, lang):
    
    if ori_ans_list == new_ans_list:
        return doc

    # print(ori_ans_list, new_ans_list)
    
    ori_ans_list = ori_ans_list[lang]
    new_ans_list = new_ans_list[lang]

    if not isinstance(ori_ans_list, list):
        ori_ans_list = [ori_ans_list]
    if not isinstance(new_ans_list, list):
        new_ans_list = [new_ans_list]
    
    
    replace_note = {"zh": ".", "ja": "・"}
    for ori_ans, new_ans in zip(ori_ans_list, new_ans_list):
        # print(ori_ans, new_ans)
        if lang not in ["zh", 'ja']:
            ori_ans = split_last_space(ori_ans)
            new_ans = split_last_space(new_ans)
            for i1, i2 in zip(ori_ans, new_ans):
                # print(i1,i2)
                doc = doc.replace(i1, i2)
        else:    
            ori_ans = split_last_tk(ori_ans, replace_note[lang])
            new_ans = split_last_tk(new_ans, replace_note[lang])
            for i1, i2 in zip(ori_ans, new_ans):
                doc = doc.replace(i1, i2)
    
    # print(doc)
    # input()

    return doc

def processdata(instance, noise_rate, chunk_num, filename, language, mode="golden", noise_doc='', contriever=None, con_tok=None, stanza_pipeline=None, tokenizer=None, languages=None, sample_languages_candidate=[], sample_num=0):
    # print(f'processdata:{languages}')
    query = instance[f'query']
    ans = instance[f'ans']

    sampled_languages = random.sample(sample_languages_candidate, k=sample_num)
    print(f'sampled_languages:{sampled_languages}')
    sample_ans_language = random.choices(sampled_languages, k=1)
    if isinstance(sample_ans_language, list):
        sample_ans_language = sample_ans_language[0]
    sampled_ans = ans[sample_ans_language]

    ans['sample'] = sampled_ans
    ans['sample_ans_language'] = sample_ans_language
    ans['sampled_languages'] = sampled_languages
    docs = instance['doc']
    for tmp_lang in sampled_languages:
        docs[tmp_lang] = doc_replace_entity(ans[tmp_lang] , ans['sample'], docs[tmp_lang], tmp_lang)
        ans[tmp_lang] = sampled_ans
    

    ret_docs = []
    doc_lang = {}
    # print(f'languages:{languages}')
    for lang in languages:
        tmp_docs = split_into_chunks(docs[lang], ans[lang], chunk_num, mode, noise_rate, noise_doc, contriever, con_tok, query[lang], lang=lang, stanza_pipeline=stanza_pipeline[lang], tokenizer=tokenizer)
        doc_lang[lang] = tmp_docs
        # ret_docs.append(tmp_docs)
    
    # if not isinstance(docs, list):
    #     docs = [docs]


    # random.shuffle(ret_docs)
    
    return query, ans, doc_lang

def predict(query, ground_truth, lang_docs, model, system, instruction, dataset, language, stanza_list=None, languages=None, pos=None, pos_lang=None):
    '''
        label: 0 for positive, 1 for negative, -1 for not enough information
    '''
    # print("DOC:", docs)
    ret_doc = []
    docs = ''
    if len(lang_docs.keys()) == 0:
        # text = instruction.format(QUERY=query, DOCS='')
        text = query
        # print(text)
        prediction = model.chat(prompt=text, temperature=0, max_tokens=256)
    else:
        if pos_lang is None or pos is None:
            ret_doc_list = []
            for k, lang_d in lang_docs.items():
                # list
                lang_d = ' ... '.join(lang_d)
                ret_doc_list.append(lang_d)
                random.shuffle(ret_doc_list)
                random.shuffle(ret_doc_list)
                random.shuffle(ret_doc_list)
        else:
            ret_doc = {}
            ret_doc_list = []
            for k, lang_d in lang_docs.items():
                # list
                lang_d = ' ... '.join(lang_d)
                ret_doc[k] = lang_d
                if k != pos_lang:
                    ret_doc_list.append(lang_d)

            random.shuffle(ret_doc_list)
            random.shuffle(ret_doc_list)
            random.shuffle(ret_doc_list)
            front_list = ret_doc_list[0:pos]
            back_list = ret_doc_list[pos:]
            if not isinstance(front_list, list):
                front_list = []
            if not isinstance(back_list, list):
                back_list = []    
            ret_doc_list = front_list + [ret_doc[pos_lang]] + back_list

        docs = ''
        for doc in ret_doc_list:
            docs += f'{doc} '

        text = instruction.format(SYSTEM=system, QUERY=query, DOCS=docs)
        # print(text)
        prediction = model.chat(prompt=text, temperature=0, max_tokens=256)
    
    if prediction is None: 
        print("prediction is None")
        quit()
    elif type(prediction) is not str:
        prediction = prediction['message']['content']

    # if language in ["zh", "ja"]:
    #     prediction = prediction.replace(" ","")
    # print(language)


    accept_sw_labels, accept_em_labels, ngram_all, prediction, factlabel = res_process(prediction, ground_truth, stanza_list, languages)

    return accept_sw_labels, accept_em_labels, ngram_all, prediction, factlabel, docs



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--modelname', type=str, default='bloomz',
        help='model name'
    )
    parser.add_argument(
        '--contriever_path', type=str, default=repo_path("models", "facebook", "mcontriever-msmarco"),
        help='model name'
    )
    parser.add_argument(
        '--dataset', type=str, default='50Q',
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
        choices=['golden','relevant','noise','hybird', "same_topic", "same_topic_hybird", "conflict", "conflict_diff", "cl_ans_text", 'pos', 'vote']
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
        '--pos', type=str, default="0",
        help="assigned position for the specific language"
    )
    parser.add_argument(
        '--pos_lang', type=str, default="en",
        help="the specific language in the specific position",
        choices=["en", "zh", "fr", "ja", "ar", "es", "pt", "ko", "query_lang"]
    )
    parser.add_argument(
        '--sample_languages_candidate', type=str, default="[]",
        help="sample_languages_candidate"
    )
    parser.add_argument(
        '--sample_num', type=str, default="0",
        help="sample num"
    )
    
    
    args = parser.parse_args()
    args.sample_num = int(args.sample_num)
    print(args.sample_languages_candidate)
    args.sample_languages_candidate = json.loads(args.sample_languages_candidate)
    candidate_num = len(args.sample_languages_candidate)

    if 'pos' not in args.mode:
        args.pos_lang = None
        args.pos = None

    languages = ["en", "zh", "fr", "ja", "ar", "es", "pt", "ko"]
    ori_languages = deepcopy(languages)

    # print(f'ori_languages:{ori_languages}')

    f1 = open(args.dataset)
    instances = json.load(f1)
    f1.close()

    noise_pool = {}

    modelname = args.modelname.split('/')[-1]

    args.dataset = args.dataset.split('/')[-1].replace('.json', '')
    
    resultpath = f'./results/multi_lingual/{args.dataset}_{args.description}/{args.mode}'
    if not os.path.exists(f'./results/multi_lingual/'):
        os.mkdir(f'./results/multi_lingual/')
    if not os.path.exists(f'./results/multi_lingual/{args.dataset}_{args.description}/'):
        os.mkdir(f'./results/multi_lingual/{args.dataset}_{args.description}/')
    if not os.path.exists(resultpath):
        os.mkdir(resultpath)

    if not os.path.exists(resultpath+f'/{modelname}'):
        os.mkdir(resultpath+f'/{modelname}')

    prompt = yaml.load(open(repo_path('instructions', 'instruction_mul.yaml'), 'r'), Loader=yaml.FullLoader)
    if args.mode == "cl_ans_text":
        prompt = yaml.load(open('./instructions/instruction_cl_db.yaml', 'r'), Loader=yaml.FullLoader)
        
    system = ''
    if args.description == 'lang_specific':
        system = prompt[args.language]['system']
    elif args.description == 'en':
        system = prompt['en_description']
        
    instruction = prompt[args.language]['instruction']

    if 'qwen' in args.modelname.lower():
        model = Qwen(model=args.modelname)
        
    filename = f'{resultpath}/{modelname}/prediction_n_sample{args.sample_num}_cnum{candidate_num}_{args.dataset}_ql{args.language}_multi_conflict_{modelname}_noise{args.noise_rate}_chunk{args.chunk_num}.json'
    
    model_tokenizer = model.get_llm_tokenizer()
    contriever = Contriever.from_pretrained(args.contriever_path)
    con_tok = AutoTokenizer.from_pretrained(args.contriever_path)
    
    stanza_lang_dict = {} 
    for lang in ori_languages:
        stanza_lang_dict[lang] = None

    results = []
    set_seed(args.seed)
    cnt = 0

    exist_line = 0
    if os.path.isfile(filename):
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():  # 如果行不是空的
                    exist_line += 1
    
    mode = 'a' if os.path.exists(filename) else 'w'

    ans_file = './data/mul_ans_all.json'
    with open(filename, mode, encoding='utf8') as f, open(ans_file) as f2:
        ans_data = json.load(f2)
        exist_cnt = 0
        for instance, ans_item in tqdm.tqdm(zip(instances, ans_data)):
            if exist_cnt < exist_line:
                exist_cnt += 1
                continue
            ans_item = ans_item['new_ans']
            instance['ans'] = ans_item
            if args.chunk_num == 0:
                query = instance[f'query']
                ans = ans_item
                docs = {}
            else:
                noise_doc = ''
                query, ans, lang_docs = processdata(instance, args.noise_rate, args.chunk_num, args.dataset, args.language, args.mode, noise_doc=noise_doc, contriever=contriever, con_tok=con_tok, stanza_pipeline=stanza_lang_dict, tokenizer=model_tokenizer, languages=ori_languages, sample_languages_candidate=args.sample_languages_candidate, sample_num=args.sample_num)

            print(f'ori_languages:{ori_languages}')
            try:
                accept_sw_labels, accept_em_labels, ngram_all, prediction, factlabel, docs = predict(query, ans, lang_docs, model, system, instruction, args.dataset, args.language, stanza_list=stanza_lang_dict, languages=ori_languages, pos=args.pos, pos_lang=args.pos_lang)
                # print(f'kori_languages:{ori_languages}')
                if not isinstance(accept_sw_labels, set) and 'sampled_languages' in  accept_sw_labels.keys():
                    languages += ['sampled_languages']
                
                newinstance = {
                    'id': cnt,
                    f'query': query[args.language],
                    f'ans': ans,
                    f'prediction': prediction,
                    f'docs': docs,
                    f'noise_rate': args.noise_rate,
                    f'factlabel': factlabel,
                    f"ngram_all": ngram_all,
                    f'accept_sw_labels': accept_sw_labels,
                    f'accept_em_labels': accept_em_labels
                }
            except Exception as e:
                print(e)
                newinstance = {
                    'id': cnt,
                    f'query': query[args.language],
                    f'ans': ans,
                    f'prediction': "PASS",
                }
            results.append(newinstance)

            # print(newinstance)
            # input()
            
            # ["en", "ar", "fr", "0"]
            
            cnt += 1
            # print(f'3ori_languages:{ori_languages}')
            f.write(json.dumps(newinstance, ensure_ascii=False)+'\n')
    
    # ["en", "ar", "fr", "0"]            
    res_dict_acc_sw = {}
    res_dict_acc_em = {}
    res_dict_acc_sw_lk = {}
    res_dict_acc_em_lk = {}
    res_dict_acc_3gram = {}
    res_dict_acc_3gram_lang = {}
 
    
    for lang in languages:
        res_dict_acc_em[lang] = 0
        res_dict_acc_sw[lang] = 0
        res_dict_acc_sw_lk[lang] = 0
        res_dict_acc_em_lk[lang] = 0
        res_dict_acc_3gram[lang] = 0
        res_dict_acc_3gram_lang[lang] = 0
    
    # print(f'4ori_languages:{ori_languages}')
    results = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            results.append(json.loads(line))
    cnt_jian = 0
    for i in results:
        if i['prediction'] == 'PASS':
            cnt_jian += 1
            continue
        accept_sw_labels = i['accept_sw_labels']
        accept_em_labels = i['accept_em_labels']
        if "NONE" in accept_sw_labels.keys():
            continue
        ngram_all = i['ngram_all']
        for lang in accept_sw_labels.keys():
            if lang == 'sampled_languages':
                continue

            if -1 not in accept_sw_labels[lang]['merge']:
                res_dict_acc_sw[lang] += sum(accept_sw_labels[lang]['merge']) / len(accept_sw_labels[lang]['merge'])
            if -1 not in accept_em_labels[lang]['merge']:
                res_dict_acc_em[lang] += sum(accept_em_labels[lang]['merge']) / len(accept_em_labels[lang]['merge'])

            res_dict_acc_3gram[lang] += ngram_all[lang]['max']
            max_lang_name = ngram_all[lang]['max_lang']
            if len(max_lang_name):
                res_dict_acc_3gram_lang[max_lang_name] += 1

            for lang2 in languages:
                if lang2 != 'sampled_languages':
                    if 1 in accept_sw_labels[lang][lang2]:
                        res_dict_acc_sw_lk[lang2] += 1
                    if 1 in accept_em_labels[lang][lang2]:
                        res_dict_acc_em_lk[lang2] += 1

    for lang in languages:
        if lang == 'sampled_languages':
            continue
        res_dict_acc_sw_lk[lang] = round(res_dict_acc_sw_lk[lang] * 100 / (len(results)-cnt_jian), 2)
        res_dict_acc_sw_lk[lang] = round(res_dict_acc_sw_lk[lang] * 100 / (len(results)-cnt_jian) , 2)
        res_dict_acc_sw_lk[lang] = round(res_dict_acc_sw_lk[lang] * 100 / (len(results)-cnt_jian) , 2) 
        res_dict_acc_em_lk[lang] = round(res_dict_acc_em_lk[lang] * 100 / (len(results)-cnt_jian) , 2)
        res_dict_acc_3gram[lang] = round(res_dict_acc_3gram[lang] * 100 / (len(results)-cnt_jian) , 2)          
        
    
    
    scores = {
    'accept_sw_labels': res_dict_acc_sw_lk,
    'accept_em_labels':  res_dict_acc_sw_lk,
    'language': args.language,
    'language mode': args.language_mode,
    'noise_rate': args.noise_rate,
    'nums': len(results)-cnt_jian,
    'res_dict_acc_sw_lk': res_dict_acc_sw_lk,
    'res_dict_acc_em_lk': res_dict_acc_em_lk,
    'res_dict_acc_3gram': res_dict_acc_3gram,
    'res_dict_acc_3gram_lang': res_dict_acc_3gram_lang
    }
    
    json.dump(scores,open(f'{resultpath}/{args.modelname}/prediction_sample{args.sample_num}_cnum{candidate_num}_{args.dataset}_ql{args.language}_lm{args.language_mode}_{args.modelname}_noise{args.noise_rate}_chunk{args.chunk_num}_seed{args.seed}_result.json','w'),ensure_ascii=False,indent=4)
    print ('Done')
