"""Deterministic, configurable severity scoring; AI cannot set score or verdict."""

from .schemas import ControllerAssessment, QualityIssue, QualityRound


def score_assessment(
    assessment: ControllerAssessment,
    weights: dict[str, int],
    threshold: int,
) -> QualityRound:
    """Deduplicate overlapping issues and enforce high/critical hard stops.

    Only medium penalties contribute to the repair threshold. Low-severity
    suggestions appear in the report but cannot independently trigger repair.
    The total displayed score contains all severities for diagnostics.
    """
    unique: list[QualityIssue] = []
    seen: set[tuple] = set()
    for issue in assessment.issues:
        key = (
            issue.type,
            issue.source_span.casefold().strip(),
            issue.translated_span.casefold().strip(),
            issue.context_rule_id,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(issue)

    score = sum(weights[issue.severity] for issue in unique)
    medium_score = sum(weights['medium'] for issue in unique if issue.severity == 'medium')
    hard_stop = any(issue.severity in ('critical', 'high') for issue in unique)
    return QualityRound(
        issues=unique,
        penalty_score=score,
        requires_repair=hard_stop or medium_score >= threshold,
    )
