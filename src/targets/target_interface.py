from abc import ABC, abstractmethod


class TargetModel(ABC):
    @abstractmethod
    def respond(self, conversation: list[dict]) -> str:
        raise NotImplementedError
