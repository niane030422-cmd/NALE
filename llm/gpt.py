import os
import time
import requests
from typing import Optional, List, Dict
from openai import OpenAI
import tiktoken

from public_repo import require_env

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

class GPT():
    
    def __init__(self, model):
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.client = OpenAI(base_url=base_url, api_key=require_env("OPENAI_API_KEY"))
        self.MAX_API_RETRY = 15
        self.LLM_RETRY_SLEEP = 5

        self.model = model
        self.tokenizer = tiktoken.get_encoding("o200k_base")

    
    def get_llm_tokenizer(self):
        return self.tokenizer   

    def chat(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
        stop: Optional[List[str]] = None,
        max_tokens: int = 8192,
        n: int = 1,
        temperature: float = 0.8,
        top_p: int = 1,
    ):

        prompt = cut_off(prompt, self.get_llm_tokenizer())

        if messages is None:
            assert isinstance(prompt, str)
            messages = [{'role': 'user', 'content': prompt}]
        else:
            assert prompt is None, 'Do not pass prompt and messages at the same time.'
        
        for i in range(self.MAX_API_RETRY):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    stream=False,
                    messages=messages
                )
                return completion.choices[0].message
            except Exception as e:
                time.sleep(self.LLM_RETRY_SLEEP)
                continue

        return None
