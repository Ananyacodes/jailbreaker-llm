from abc import ABC, abstractmethod


class TargetModel(ABC):
    @abstractmethod
    def respond(self, conversation: list[dict]) -> str:
        raise NotImplementedError

    def respond_many(self, conversations: list[list[dict]]) -> list[str]:
        """Respond to several conversations.

        Remote/custom targets keep the safe sequential behaviour by default.
        Local targets override this to use a single padded model batch.
        """
        return [self.respond(conversation) for conversation in conversations]
