"""One bounded Translate→Critique→Repair→Verify quality workflow."""

import json

from app.config import Settings
from app.contexts.repository import ResolvedContext
from app.services.openai import translate_values

from .controller import review, require_independent_controller
from .schemas import QualityReport
from .scoring import score_assessment


class VerificationFailed(RuntimeError):
    """The corrected translation still fails the configured quality gate."""

    def __init__(self, report: QualityReport):
        self.report = report
        super().__init__('Translation did not pass independent verification')


def _report(status, controller_id, context, first, second=None, repairs=0):
    return QualityReport(
        status=status, controller_model=controller_id,
        context_id=context.context_id if context else None,
        context_version=context.version if context else None,
        repair_attempts=repairs, initial=first, recheck=second,
    )


async def verify_translation(
    original: list[str], translated: list[str], target_language: str,
    settings: Settings, context: ResolvedContext | None,
    context_prompt: str | None,
) -> tuple[list[str], QualityReport]:
    """At most one repair and two controller evaluations; fail closed on FAIL."""
    controller_id = require_independent_controller(settings)
    threshold = settings.localizer_qa_repair_threshold
    weights = settings.localizer_qa_penalty_weights
    initial = score_assessment(
        await review(original, translated, target_language, settings, context),
        weights, threshold,
    )
    if not initial.requires_repair:
        return translated, _report('pass', controller_id, context, initial)

    issues = [item.model_dump() for item in initial.issues]
    repair_hint = (
        (context_prompt or '')
        + '\nThe previous draft translation had these specific issues. '
        + 'Previous translated array: ' + json.dumps(translated, ensure_ascii=False)
        + '\nController issues: ' + json.dumps(issues, ensure_ascii=False)
        + '\nCorrect the listed issues while preserving already-correct segments. '
        + 'Return ONLY the corrected translated JSON array, in the original order.'
    )
    repaired = await translate_values(
        original, target_language, settings, context_prompt=repair_hint,
    )
    second = score_assessment(
        await review(original, repaired, target_language, settings, context),
        weights, threshold,
    )
    if second.requires_repair:
        raise VerificationFailed(_report(
            'verification_failed', controller_id, context, initial, second, repairs=1,
        ))
    return repaired, _report('pass', controller_id, context, initial, second, repairs=1)
