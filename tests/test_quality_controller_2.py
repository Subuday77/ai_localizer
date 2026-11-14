"""Offline verified QA contract and strict translator/controller separation tests."""

import asyncio
import hashlib

import httpx
import pytest

from app.config import Settings, get_settings
from app.main import app
from app.quality import pipeline
from app.quality.controller import (
    QualityConfigurationError, require_independent_controller,
)
from app.quality.pipeline import VerificationFailed
from app.quality.schemas import ControllerAssessment, QualityIssue, QualityReport, QualityRound
from app.quality.scoring import score_assessment


def settings(**overrides):
    values = dict(
        _env_file=None,
        openai_model_id='translator-a',
        openai_fallback_model_id='translator-fallback',
        localizer_qa_verified_enabled=True,
        localizer_qa_controller_model_id='controller-b',
        # Explicit *test* fixture weights, not production defaults.
        localizer_qa_penalty_weights={
            'critical': 40, 'high': 20, 'medium': 6, 'low': 1,
        },
        localizer_qa_repair_threshold=12,
    )
    values.update(overrides)
    return Settings(**values)


def issue(severity, reason='Example issue'):
    return QualityIssue(
        type='terminology', severity=severity,
        source_span='conductor', translated_span='проводник', reason=reason,
    )


def test_controller_must_differ_from_both_translator_and_fallback():
    assert require_independent_controller(settings()) == 'controller-b'
    for model in ('translator-a', 'translator-fallback'):
        with pytest.raises(QualityConfigurationError, match='differ'):
            require_independent_controller(settings(localizer_qa_controller_model_id=model))
    with pytest.raises(QualityConfigurationError, match='disabled'):
        require_independent_controller(settings(localizer_qa_verified_enabled=False))
    with pytest.raises(QualityConfigurationError, match='all severity'):
        require_independent_controller(settings(localizer_qa_penalty_weights={}))
    with pytest.raises(QualityConfigurationError, match='threshold'):
        require_independent_controller(settings(localizer_qa_repair_threshold=None))


def test_severity_scoring_and_highest_duplicate_wins():
    policy = settings()
    weights = policy.localizer_qa_penalty_weights
    threshold = policy.localizer_qa_repair_threshold
    low = score_assessment(ControllerAssessment(issues=[issue('low')]), weights, threshold)
    assert low.penalty_score == 1 and not low.requires_repair
    two_medium = score_assessment(
        ControllerAssessment(issues=[
            issue('medium'),
            QualityIssue(type='style', severity='medium', reason='Wrong tone'),
        ]), weights, threshold,
    )
    assert two_medium.penalty_score == 12 and two_medium.requires_repair
    duplicate = score_assessment(
        ControllerAssessment(issues=[issue('low'), issue('high')]),
        weights, threshold,
    )
    assert len(duplicate.issues) == 1
    assert duplicate.issues[0].severity == 'high'
    assert duplicate.penalty_score == 20 and duplicate.requires_repair
    assert score_assessment(
        ControllerAssessment(issues=[issue('critical')]), weights, 1000,
    ).requires_repair


def test_verified_pass_with_only_low_issue_has_no_repair(monkeypatch):
    reviews = []

    async def fake_review(*args):
        reviews.append(args)
        return ControllerAssessment(issues=[issue('low')])

    async def unexpected_translate(*args, **kwargs):
        raise AssertionError('Repair should not be invoked')

    monkeypatch.setattr(pipeline, 'review', fake_review)
    monkeypatch.setattr(pipeline, 'translate_values', unexpected_translate)
    result, report = asyncio.run(pipeline.verify_translation(
        ['Hello'], ['Привет'], 'Russian', settings(), None, None,
    ))
    assert result == ['Привет']
    assert report.status == 'pass' and report.repair_attempts == 0
    assert report.initial.issues[0].severity == 'low'
    assert len(reviews) == 1
    assert reviews[0][0] == ['Hello'] and reviews[0][1] == ['Привет']


def test_high_issue_triggers_exactly_one_repair_and_recheck(monkeypatch):
    assessments = [ControllerAssessment(issues=[issue('high')]), ControllerAssessment(issues=[])]
    repaired_prompts = []

    async def fake_review(*args):
        return assessments.pop(0)

    async def fake_translate(original, language, config, context_prompt=None):
        repaired_prompts.append(context_prompt)
        assert original == ['The conductor checked tickets.']
        return ['Кондуктор проверил билеты.']

    monkeypatch.setattr(pipeline, 'review', fake_review)
    monkeypatch.setattr(pipeline, 'translate_values', fake_translate)
    result, report = asyncio.run(pipeline.verify_translation(
        ['The conductor checked tickets.'], ['Проводник проверил билеты.'],
        'Russian', settings(), None, 'Railway context',
    ))
    assert result == ['Кондуктор проверил билеты.']
    assert report.status == 'pass' and report.repair_attempts == 1
    assert report.initial.requires_repair
    assert not report.recheck.requires_repair
    assert len(repaired_prompts) == 1
    assert 'Railway context' in repaired_prompts[0]
    assert 'Проводник проверил билеты.' in repaired_prompts[0]
    assert not assessments


def test_repair_that_still_fails_raises_with_complete_report(monkeypatch):
    calls = {'controller': 0, 'translator': 0}

    async def always_bad(*args):
        calls['controller'] += 1
        return ControllerAssessment(issues=[issue('high')])

    async def repair_once(*args, **kwargs):
        calls['translator'] += 1
        return ['Still incorrect']

    monkeypatch.setattr(pipeline, 'review', always_bad)
    monkeypatch.setattr(pipeline, 'translate_values', repair_once)
    with pytest.raises(VerificationFailed) as failure:
        asyncio.run(pipeline.verify_translation(
            ['Source'], ['Bad'], 'Russian', settings(), None, None,
        ))
    assert failure.value.report.status == 'verification_failed'
    assert failure.value.report.recheck.requires_repair
    assert failure.value.report.repair_attempts == 1
    assert calls == {'controller': 2, 'translator': 1}


def api(json_body, mode_settings):
    from app.api import routes

    token = 'qa-test-token-not-real'
    digest = hashlib.sha256(token.encode()).hexdigest()
    mode_settings = mode_settings.model_copy(update={
        'localizer_require_api_key': True,
        'localizer_api_key_hashes': [digest],
    })
    app.dependency_overrides[get_settings] = lambda: mode_settings

    async def call():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url='http://test',
        ) as client:
            return await client.post('/translate', json=json_body,
                                     headers={'Authorization': f'Bearer {token}'})
    try:
        return asyncio.run(call())
    finally:
        app.dependency_overrides.clear()


def test_verified_is_disabled_by_default_and_never_silently_standard():
    r = api({
        'dictionary': {'a': 'Hello'}, 'language_code': 'de',
        'quality_mode': 'verified',
    }, settings(localizer_qa_verified_enabled=False))
    assert r.status_code == 403


def test_verified_api_returns_report_on_success(monkeypatch):
    from app.api import routes

    async def fake_translation(values, language, config):
        return ['Hallo']

    async def fake_pipeline(original, translated, target, config, context, context_hint):
        return translated, QualityReport(
            status='pass', controller_model='controller-b', repair_attempts=0,
            initial=QualityRound(issues=[], penalty_score=0, requires_repair=False),
        )

    monkeypatch.setattr(routes, 'translate_values', fake_translation)
    monkeypatch.setattr(routes, 'verify_translation', fake_pipeline)
    r = api({
        'dictionary': {'a': 'Hello'}, 'language_code': 'de',
        'quality_mode': 'verified',
    }, settings())
    assert r.status_code == 200, r.text
    assert r.json()['dictionary'] == {'a': 'Hallo'}
    assert r.json()['quality_report']['status'] == 'pass'
    assert r.json()['quality_report']['controller_model'] == 'controller-b'


def test_verified_api_fails_closed_on_recheck_error(monkeypatch):
    from app.api import routes

    async def fake_translation(values, language, config):
        return ['Wrong']

    async def failed_pipeline(*args):
        raise VerificationFailed(QualityReport(
            status='verification_failed', controller_model='controller-b',
            repair_attempts=1,
            initial=QualityRound(issues=[issue('high')], penalty_score=20, requires_repair=True),
            recheck=QualityRound(issues=[issue('high')], penalty_score=20, requires_repair=True),
        ))

    monkeypatch.setattr(routes, 'translate_values', fake_translation)
    monkeypatch.setattr(routes, 'verify_translation', failed_pipeline)
    r = api({
        'dictionary': {'a': 'Hello'}, 'language_code': 'de',
        'quality_mode': 'verified',
    }, settings())
    assert r.status_code == 422
    assert r.json()['detail']['status'] == 'verification_failed'
    assert r.json()['detail']['quality_report']['repair_attempts'] == 1
    assert 'dictionary' not in r.json()


def test_verified_api_does_not_fallback_to_unverified_text(monkeypatch):
    from app.api import routes

    async def broken_translation(*args):
        raise RuntimeError('provider timeout')

    monkeypatch.setattr(routes, 'translate_values', broken_translation)
    r = api({
        'dictionary': {'a': 'Hello'}, 'language_code': 'de',
        'quality_mode': 'verified',
    }, settings())
    assert r.status_code == 503
    assert 'dictionary' not in r.json()
