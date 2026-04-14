from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time


class ActionStatus(str, Enum):
    SUCCESS = "success"
    FAILED  = "failed"
    SKIPPED = "skipped"   # e.g. cooldown active


@dataclass
class ActionResult:
    action_name: str
    status:      ActionStatus
    message:     str = ""
    payload:     Optional[dict] = None
    timestamp:   float = field(default_factory=time.time)
