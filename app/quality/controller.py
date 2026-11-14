"""Independent AI reviewer: sees source, translation, language and exact context."""

import json

from pydantic import ValidationError

from app.config import Settings
from app.contexts.repository import ResolvedContext
from app.services.openai import _load_model_json, _request_with_retries

from .schemas import ControllerAssessment


class QualityConfigurationError(RuntimeError):
    pass


def require_independent_controller(settings: Settings) -> str:
    """Guarantee a controller ID distinct from every configured translator model."""
    controller = settings.localizer_qa_controller_model_id.strip()
    if not settings.localizer_qa_verified_enabled:
        raise QualityConfigurationError('Verified mode is disabled')
    if not settings.localizer_qa_verified_api_key_hashes:
        raise QualityConfigurationError('Verified API key grants are not configured')
    if not controller or controller.startswith('your-'):
        raise QualityConfigurationError('Controller model ID is not configured')
    if controller in {settings.openai_model_id, settings.openai_fallback_model_id}:
        raise QualityConfigurationError('Controller model must differ from translator AND fallback')
    if set(settings.localizer_qa_penalty_weights) != {'critical', 'high', 'medium', 'low'}:
        raise QualityConfigurationError('Configure all severity weights before enabling verified')
    if any(type(value) is not int or value < 0 for value in settings.localizer_qa_penalty_weights.values()):
        raise QualityConfigurationError('Severity weights must be nonnegative integers')
    if not settings.localizer_qa_repair_threshold or settings.localizer_qa_repair_threshold <= 0:
        raise QualityConfigurationError('Repair threshold is not configured')
    return controller


def _build_review_prompt(
    original: list[str], translated: list[str], target_language: str,
    context: ResolvedContext | None,
) -> str:
    """Request evidence-based issues, not an AI-provided final verdict."""
    if len(original) != len(translated):
        raise ValueError('Review candidate count mismatch')
    context_data = None
    if context is not None:
        context_data = {
            'id': context.context_id,
            'version': context.version,
            'rules': [r.model_dump() for r in context.rules],
            'glossary': [g.model_dump() for g in context.glossary],
            'style': context.style,
        }
    payload = {
        'source_language': 'en', 'target_language': target_language,
        'source_strings': original, 'translated_strings': translated,
        'context': context_data,
    }
    return (
        'You are an independent translation quality controller. Assess whether '
        'each target-language string faithfully conveys its corresponding source '
        'string. Evaluate meaning, omissions/additions, terminology, the semantic '
        'applicability of context rules, tone and placeholders. Context is a hint; '
        'do NOT demand a terminology substitution that contradicts an unambiguous '
        'source sentence. Report only concrete errors, not invented issues. '
        'Output ONLY valid JSON in this exact shape: {"issues":[{"type":"meaning",'
        '"severity":"high","source_span":"...","translated_span":"...",'
        '"expected":"...","context_rule_id":null,"reason":"..."}]}. '
        'Allowed type: meaning, omission, addition, context_mismatch, '
        'terminology, style, format, other. Allowed severity: critical, high, '
        'medium, low. For no issues return {"issues":[]}. Maximum 50 issues. '
        'Use strings for all fields, except context_rule_id may be null. '
        'Review payload: ' + json.dumps(payload, ensure_ascii=False)
    )


async def review(
    original: list[str], translated: list[str], target_language: str,
    settings: Settings, context: ResolvedContext | None,
) -> ControllerAssessment:
    controller_id = require_independent_controller(settings)
    controller_settings = settings.model_copy(update={
        'openai_model_id': controller_id,
        'openai_fallback_model_id': '',
    })
    prompt = _build_review_prompt(original, translated, target_language, context)

    def parse(content: str) -> ControllerAssessment:
        result = _load_model_json(content)
        return ControllerAssessment.model_validate(result)

    return await _request_with_retries(
        controller_settings,
        prompt_builder=lambda retry: prompt + (
            '\\nPrevious review was invalid JSON; return the same assessment '
            'as a valid JSON object without extra keys.' if retry else ''
        ),
        parser=parse,
        operation='quality review',
    )
