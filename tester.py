from tqdm.auto import trange
import urllib.parse
import requests
import random
import numpy as np
import io

def get_random_string():
    return ''.join(random.choices('abcdefghijklmnopqrstuvwxyz ', k=40))


model_name = "mixedbread-ai/mxbai-embed-large-v1"
for _ in trange(1000):
    data = dict(input=[get_random_string() for _ in range(20)])
    print(urllib.parse.quote_plus(model_name))
    result = requests.post(f"http://127.0.0.1:7000/{urllib.parse.quote_plus(model_name)}", json=data)
    print(result.status_code, len(result.content))