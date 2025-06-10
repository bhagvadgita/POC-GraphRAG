import json
import os
from typing import Dict, Any, Optional
from .base import BaseGraphStorage, BaseVectorStorage, BaseKVStorage

DEFAULT_STORAGE_PATH = os.path.join(os.getcwd(), "graphrag","data", "storage")

class LocalGraphStorage(BaseGraphStorage):
    def __init__(self, storage_path: str = DEFAULT_STORAGE_PATH):
        self.storage_path = storage_path
        self.graph_file = os.path.join(storage_path, "graph.json")
        os.makedirs(storage_path, exist_ok=True)
        if not os.path.exists(self.graph_file):
            with open(self.graph_file, 'w') as f:
                json.dump({"nodes": {}, "edges": {}}, f)

    async def upsert(self, data: Dict[str, Any]) -> None:
        with open(self.graph_file, 'r+') as f:
            graph_data = json.load(f)
            # Merge new data with existing data
            for key, value in data.items():
                if key in graph_data:
                    graph_data[key].update(value)
                else:
                    graph_data[key] = value
            f.seek(0)
            json.dump(graph_data, f, indent=2)
            f.truncate()

class LocalVectorStorage(BaseVectorStorage):
    def __init__(self, storage_path: str = DEFAULT_STORAGE_PATH):
        self.storage_path = storage_path
        self.vector_file = os.path.join(storage_path, "vectors.json")
        os.makedirs(storage_path, exist_ok=True)
        if not os.path.exists(self.vector_file):
            with open(self.vector_file, 'w') as f:
                json.dump({}, f)

    async def upsert(self, data: Dict[str, Any]) -> None:
        with open(self.vector_file, 'r+') as f:
            vector_data = json.load(f)
            vector_data.update(data)
            f.seek(0)
            json.dump(vector_data, f, indent=2)
            f.truncate()

class LocalKVStorage(BaseKVStorage):
    def __init__(self, storage_path: str = DEFAULT_STORAGE_PATH):
        self.storage_path = storage_path
        self.kv_file = os.path.join(storage_path, "kv_store.json")
        os.makedirs(storage_path, exist_ok=True)
        if not os.path.exists(self.kv_file):
            with open(self.kv_file, 'w') as f:
                json.dump({}, f)

    async def get(self, key: str) -> Optional[Any]:
        with open(self.kv_file, 'r') as f:
            kv_data = json.load(f)
            return kv_data.get(key)

    async def set(self, key: str, value: Any) -> None:
        with open(self.kv_file, 'r+') as f:
            kv_data = json.load(f)
            kv_data[key] = value
            f.seek(0)
            json.dump(kv_data, f, indent=2)
            f.truncate() 