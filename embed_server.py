import io
import os
import logging
from typing import List, Iterable

import numpy as np
import torch

from fastapi import BackgroundTasks, FastAPI
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from functools import lru_cache
from more_itertools import chunked
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from cache import SQLiteCacheBackend, CacheBackend


# todo: autoboot and scale to zero


class EncodeRequest(BaseModel):
    input: List[str]


app = FastAPI()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class E5Model:
    @staticmethod
    def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def __init__(self, model, device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if device == "cpu":
            self.model = AutoModel.from_pretrained(model)
        else:
            self.model = AutoModel.from_pretrained(model)
            self.model = self.model.to(device)
        self.device = device

    def _encode(self, input_texts: List[str]) -> np.array:
        batch_dict = self.tokenizer(
            [
                "query: " + s
                if not (s.startswith("query: ") or s.startswith("passage: "))
                else s
                for s in input_texts
            ],
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}

        outputs = self.model(**batch_dict)
        embeddings = self.average_pool(
            outputs.last_hidden_state, batch_dict["attention_mask"]
        )
        return embeddings.detach().cpu().numpy()

    def encode(self, input_texts: List[str], batch_size=8):
        result = []
        for chunk in chunked(input_texts, batch_size):
            result.extend(self._encode(chunk))
        return np.array(result)


def initialize_model(model, device):
    if model.startswith("sentence-transformers/") or model == 'mixedbread-ai/mxbai-embed-large-v1':
        return SentenceTransformer(model, device=device)
    elif "e5-" in model:
        return E5Model(model, device)


def load_model_cpu(model):
    logger.info(f"Loading model {model} on CPU")
    return initialize_model(model, "cpu")


def load_model(model):
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    if os.path.exists("/tmp/.use-cpu"):
        logger.info(f"Force using CPU mode due to /tmp/.use-cpu file")
        return load_model_cpu(model)
    else:
        try:
            logger.info(f"Loading model {model} using device: {device}")
            return initialize_model(model, device)
        except Exception as exc:
            import traceback
            traceback.print_tb(exc)
            logger.warning(f"Failed to load model on {device}, falling back to CPU mode")
            return load_model_cpu(model)


models = {
    "sentence-transformers/all-mpnet-base-v2": None,
    "intfloat/e5-large-v2": None,
    "mixedbread-ai/mxbai-embed-large-v1": None
}

cache_backend: CacheBackend = SQLiteCacheBackend("embeddings_cache")

# Log available devices at startup
if torch.cuda.is_available():
    logger.info(f"CUDA is available: {torch.cuda.get_device_name()}")
elif torch.backends.mps.is_available():
    logger.info("Apple M-series GPU (MPS) is available")
else:
    logger.info("No GPU detected, will use CPU")

logger.info("Models and cache ready")


def yield_from_file(file):
    yield file.getvalue()


def add_to_cache(cache_backend, transformer, strings, embeddings):
    pairs = []
    for string, embedding in zip(strings, embeddings):
        virt_file = io.BytesIO()
        np.save(virt_file, embedding, allow_pickle=False)  # type: ignore
        virt_file.seek(0)
        pairs.append(
            (transformer.encode("utf-8") + string.encode("utf-8"), virt_file.read())
        )
    cache_backend.write_batch(pairs)


@app.post("/{transformer:path}")
async def root(
    transformer, encode_request: EncodeRequest, background_tasks: BackgroundTasks
):
    logger.info(f"Received encoding request for transformer '{transformer}' with {len(encode_request.input)} inputs")
    # fixme: support sending as binary data
    if models[transformer] is None:
        logger.info(f"Loading model {transformer}")
        models[transformer] = load_model(transformer)
    if transformer in (
        'text-embedding-ada-002',
        'text-embedding-3-small',
        'text-embedding-3-large'
    ):
        def encode(data: Iterable[str]):
            from embedapi import _openai_encode_batch
            return _openai_encode_batch(transformer, data)
    else:
        import torch
        encode = torch.no_grad(models[transformer].encode)
    # find cached embeddings
    # perf: would using a pre-allocated numpy array of arrays be faster?
    to_embed = []
    embeddings = []
    empty_indexes = []
    for index, string in enumerate(encode_request.input):
        cached_data = cache_backend.get(
            transformer.encode("utf-8") + string.encode("utf-8")
        )
        if cached_data is None:
            to_embed.append(string)
            empty_indexes.append(index)
            embeddings.append(None)
        else:
            # perf: this loads binary->numpy->binary; one solution would be to return a JSON array instead of a numpy one
            embeddings.append(np.load(io.BytesIO(cached_data), allow_pickle=False))
    if len(to_embed) > 0:
        logger.info(f"Encoding {len(to_embed)} uncached inputs")
        try:
            encoded_embeddings = encode(tuple(to_embed))
        finally:
            import torch.cuda
            torch.cuda.empty_cache()
        for i, (string, embedding) in enumerate(zip(to_embed, encoded_embeddings)):
            embeddings[empty_indexes[i]] = embedding
        background_tasks.add_task(
            add_to_cache, cache_backend, transformer, to_embed, encoded_embeddings
        )
    virt_file = io.BytesIO()
    np.save(virt_file, embeddings, allow_pickle=False)  # type: ignore
    logger.info(f"Dispatching response with {len(embeddings)} embeddings")
    return StreamingResponse(yield_from_file(virt_file))
