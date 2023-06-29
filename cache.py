from abc import ABC, abstractmethod
from typing import List, Tuple

import msgpack
from sqlitedict import SqliteDict


class CacheBackend(ABC):
    @abstractmethod
    def __init__(self, path):
        pass

    @abstractmethod
    def write(self, key: bytes, value: bytes):
        pass

    def write_batch(self, key_value_pairs: List[Tuple[bytes, bytes]]):
        for key, value in key_value_pairs:
            self.write(key, value)

    @abstractmethod
    def read(self, key: bytes) -> bytes:
        pass

    def read_batch(self, keys: List[bytes]) -> List[bytes]:
        return [self.read(key) for key in keys]

    def get(self, key: bytes, default_value: bytes = None) -> bytes:
        try:
            return self.read(key)
        except KeyError:
            return default_value


class SQLiteCacheBackend(CacheBackend):
    def __init__(self, path_prefix):
        # perf: separate tables for each model
        self.db = SqliteDict(
            path_prefix + ".sqlite",
            tablename="embeddings",
            encode=msgpack.packb,
            decode=msgpack.unpackb,
            encode_key=lambda key: key,
            decode_key=lambda value: value,
        )

    def write(self, key: bytes, value: bytes):
        self.db[key] = value
        self.db.commit(blocking=False)

    def write_batch(self, key_value_pairs: List[Tuple[bytes, bytes]]):
        # perf: sorting keys may be faster
        for key, value in key_value_pairs:
            self.db[key] = value
        self.db.commit(blocking=False)

    def read(self, key: bytes) -> bytes:
        return self.db[key]

    # sqlitedict implements get() through catching a KeyError

# perf: leveldb may be faster
