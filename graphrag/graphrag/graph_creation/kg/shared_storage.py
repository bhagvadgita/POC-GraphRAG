import asyncio
from typing import Optional

class GraphDBLock:
    def __init__(self, enable_logging: bool = True):
        self._lock = asyncio.Lock()
        self.enable_logging = enable_logging

    async def __aenter__(self):
        await self._lock.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()

_graph_db_lock: Optional[GraphDBLock] = None

def get_graph_db_lock(enable_logging: bool = True) -> GraphDBLock:
    global _graph_db_lock
    if _graph_db_lock is None:
        _graph_db_lock = GraphDBLock(enable_logging)
    return _graph_db_lock 