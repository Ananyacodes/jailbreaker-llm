"""Every target model -- local HF checkpoint or hosted API -- implements
this same interface, so the attacker/agent code never needs to know which."""
from abc import ABC, abstractmethod


class TargetModel(ABC):
    @abstractmethod
    def respond(self, conversation: list[dict]) -> str:
        """conversation: list of {"role": "user"|"assistant", "content": str}
        Returns the target's next response as plain text."""
        raise NotImplementedError
