from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers.generation import GenerationConfig
from typing import Dict, Iterator, List, Optional, Union
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
import os
from openai import OpenAI
from vllm import LLM, SamplingParams

from public_repo import model_path_from_env, require_env

def cut_off(text, tokenizer, max_len=8192):

    tokens = tokenizer.encode(text)
    if len(tokens) <= max_len:
        return text

    str_query = '...\n\n'.split(text)[-1]
    str_doc = text[:-len(str_query)]

    tok_query = tokenizer.encode(str_query)
    tok_doc = tokenizer.encode(str_doc)

    tok_query_len = len(tok_query)
    tok_doc_len = len(tok_doc)

    max_token_len = max_len - tok_query_len - 1
    tok_doc = tok_doc[:max_token_len]


    truncated_text = tokenizer.decode(tok_doc+tok_query)
    return truncated_text 

class Aya():
    def __init__(self, model):
        self.model_path = model_path_from_env("AYA_MODEL_ROOT", "models", "google", model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        self.model = AutoModelForCausalLM.from_pretrained(self.model_path,
            device_map=None, trust_remote_code=True).to("cuda:0")
    
    def get_llm_tokenizer(self):
        return self.tokenizer

    def chat(self, prompt, temperature=0.8, max_tokens=8192):
        prompt = cut_off(prompt, self.tokenizer)
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.model.device)

        do_sample = True
        if temperature == 0:
            do_sample = False
        with torch.no_grad():
            gen_tokens = self.model.generate(
                input_ids, 
                max_new_tokens=max_tokens, 
                do_sample=do_sample, 
                temperature=temperature,
                )

            gen_text = self.tokenizer.decode(gen_tokens[0], skip_special_tokens=True).split('<|CHATBOT_TOKEN|>')[-1].split('<|END_OF_TURN_TOKEN|>')[0]
            return gen_text

class Qwen:
    def __init__(self, model):
        self.model_path = model_path_from_env("QWEN_MODEL_ROOT", "models", "Qwen", model)
        self.llm = LLM(
            model=self.model_path,
            trust_remote_code=True,
            dtype="auto",
            max_model_len=8192,
            gpu_memory_utilization=0.5
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

    def get_llm_tokenizer(self):
        return self.tokenizer

    def chat(self, prompt, temperature=0.8, max_tokens=8192):
        prompt = cut_off(prompt, self.tokenizer)
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        sp = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        out = self.llm.generate([text], sp)[0].outputs[0].text
        return out.strip()
