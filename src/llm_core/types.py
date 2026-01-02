from dataclasses import dataclass
from typing import Optional

# Placeholder for future generic types to decouple from provider specific types
# e.g. GenericMessage, GenericRole, etc.

@dataclass
class LLMConfig:
    temperature: float = 1.0
    max_tokens: int = 100
    system_instruction: Optional[str] = None
