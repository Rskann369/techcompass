"""TechCompass — 本地语义 Embedding。

当前模型：
    BAAI/bge-small-en-v1.5

特点：
- 本地推理，不依赖外部 API
- 模型约 130MB
- 输出 384 维向量
- 默认 L2 normalize
- 首次运行需要联网下载模型，之后可离线使用

设计：
- 进程内单例，避免重复加载模型
- 支持单条和批量 embedding
- 与 HashEmbedder 保持兼容接口
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np

SEMANTIC_MODEL_NAME = "BAAI/bge-small-en-v1.5"
SEMANTIC_MODEL_DIM = 384

# 模型已下载到项目本地，运行时完全离线。
LOCAL_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "bge-small-en-v1.5"


PROJECT_HF_HOME = Path(__file__).resolve().parents[2] / ".cache" / "huggingface"


@lru_cache(maxsize=1)
def _load_model():
    from sentence_transformers import SentenceTransformer

    # 优先加载项目内模型，完全离线；若本地模型不存在则回退 Hub 名称。
    model_path = str(LOCAL_MODEL_PATH) if LOCAL_MODEL_PATH.exists() else SEMANTIC_MODEL_NAME
    return SentenceTransformer(model_path)


class SemanticEmbedder:
    """BGE 本地语义 embedding。"""

    model_name = SEMANTIC_MODEL_NAME
    dim = SEMANTIC_MODEL_DIM

    def __init__(self, device: str | None = None):
        self._device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = _load_model()
        return self._model

    def embed(self, text: str) -> np.ndarray:
        vector = self.embed_batch([text])[0]
        return vector

    def embed_batch(self, texts: Iterable[str]) -> list[np.ndarray]:
        texts = list(texts)
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return [np.asarray(v, dtype=np.float32) for v in vectors]

    def embed_list(self, texts: Iterable[str]) -> list[list[float]]:
        return [v.tolist() for v in self.embed_batch(texts)]
