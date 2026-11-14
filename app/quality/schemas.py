"""Controller output schema and server-generated quality reports."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Severity = Literal['critical', 'high', 'medium', 'low']
IssueType = Literal[
    'meaning', 'omission', 'addition', 'context_mismatch', 'terminology',
    'style', 'format', 'other',
]


class QualityIssue(BaseModel):
    model_config = ConfigDict(extra='forbid')

    type: IssueType
    severity: Severity
    source_span: str = Field(default='', max_length=500)
    translated_span: str = Field(default='', max_length=500)
    expected: str = Field(default='', max_length=500)
    context_rule_id: str | None = Field(default=None, max_length=100)
    reason: str = Field(min_length=1, max_length=1000)


class ControllerAssessment(BaseModel):
    """The model reports issues, never an authoritative PASS or penalty score."""

    model_config = ConfigDict(extra='forbid')
    issues: list[QualityIssue] = Field(default_factory=list, max_length=50)


class QualityRound(BaseModel):
    issues: list[QualityIssue]
    penalty_score: int = Field(ge=0)
    requires_repair: bool


class QualityReport(BaseModel):
    status: Literal['pass', 'verification_failed']
    controller_model: str
    context_id: str | None = None
    context_version: int | None = None
    repair_attempts: int = Field(ge=0, le=1)
    initial: QualityRound
    recheck: QualityRound | None = None
