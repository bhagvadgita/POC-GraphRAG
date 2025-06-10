from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseGraphStorage(ABC):
    @abstractmethod
    async def upsert(self, data: Dict[str, Any]) -> None:
        pass

class BaseVectorStorage(ABC):
    @abstractmethod
    async def upsert(self, data: Dict[str, Any]) -> None:
        pass

class BaseKVStorage(ABC):
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        pass

    @abstractmethod
    async def set(self, key: str, value: Any) -> None:
        pass 