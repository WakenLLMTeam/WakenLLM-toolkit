from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List
import time


class SeverityTier(IntEnum):
    NONE     = 0
    MILD     = 1
    MODERATE = 2
    SEVERE   = 3


@dataclass
class ModalityScore:
    modality:     str
    score:        float        # 0.0 (alert) - 1.0 (severely fatigued)
    rationale:    str
    signals_used: List[str] = field(default_factory=list)


@dataclass
class JudgeVerdict:
    composite_score: float                        # weighted aggregate
    severity_tier:   SeverityTier
    modality_scores: Dict[str, ModalityScore]
    reasoning:       str
    timestamp:       float = field(default_factory=time.time)
    context_tags:    List[str] = field(default_factory=list)  # e.g. ["highway", "night"]
