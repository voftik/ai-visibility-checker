from __future__ import annotations

import json
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.models import Base, RecoveryEpoch, Run, RunStatus
from app.services.openrouter import ChatResult
from app.services.recovery_orchestrator import (
    ACTION_DETERMINISTIC_FALLBACK,
    ACTION_RETRY_WITH_GUIDANCE,
    ACTION_TARGETED_ANNOTATION_REPAIR,
    ORCHESTRATOR_MODEL,
    ORCHESTRATOR_VERSION,
    OrchestratorContractError,
    OrchestratorResult,
    plan_recovery,
    validate_recovery_decision,
)
from app.services.recovery_state import (
    RecoveryBudgetExceeded,
    finish_recovery,
    mark_recovery_executing,
    plan_durable_recovery,
    recovery_failure_fingerprint,
    recovery_scope_digest,
    stable_digest,
)
from app.services.run_lease import RunLeaseLostError, bind_run_lease


def _chat_result(parsed: dict) -> ChatResult:
    return ChatResult(
        text="{}",
        parsed=parsed,
        citations=[],
        usage={"prompt_tokens": 100},
        annotations=[],
        request_policy={},
        web_attestation={"metric_eligible": True},
        router_metadata={},
    )


class RecoveryDecisionContractTests(unittest.TestCase):
    def test_failure_fingerprint_is_stable_and_stage_specific(self) -> None:
        first = recovery_failure_fingerprint(
            stage_key="scenario_design",
            failure_class="repairable_semantic",
            failure_code="prompt_set_non_convergent",
            diagnostics={"errors": ["x"]},
        )
        second = recovery_failure_fingerprint(
            stage_key="scenario_design",
            failure_class="repairable_semantic",
            failure_code="prompt_set_non_convergent",
            diagnostics={"errors": ["x"]},
        )
        other = recovery_failure_fingerprint(
            stage_key="knowledge_gap",
            failure_class="repairable_semantic",
            failure_code="prompt_set_non_convergent",
            diagnostics={"errors": ["x"]},
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertEqual(len(stable_digest({"a": 1})), 64)

    def test_durable_scope_digest_binds_every_executable_boundary(self) -> None:
        base = recovery_scope_digest(
            facts={"profile": "same"},
            allowed_actions={ACTION_DETERMINISTIC_FALLBACK},
            permitted_answer_ids={41},
            permitted_artifact_keys={"prompt_set"},
        )
        variants = (
            recovery_scope_digest(
                facts={"profile": "same"},
                allowed_actions={ACTION_RETRY_WITH_GUIDANCE},
                permitted_answer_ids={41},
                permitted_artifact_keys={"prompt_set"},
            ),
            recovery_scope_digest(
                facts={"profile": "same"},
                allowed_actions={ACTION_DETERMINISTIC_FALLBACK},
                permitted_answer_ids={42},
                permitted_artifact_keys={"prompt_set"},
            ),
            recovery_scope_digest(
                facts={"profile": "same"},
                allowed_actions={ACTION_DETERMINISTIC_FALLBACK},
                permitted_answer_ids={41},
                permitted_artifact_keys={"semantic_review"},
            ),
        )
        self.assertTrue(all(value != base for value in variants))

    def test_decision_cannot_touch_unlisted_answers_or_artifacts(self) -> None:
        with self.assertRaises(OrchestratorContractError):
            validate_recovery_decision(
                {
                    "action": ACTION_TARGETED_ANNOTATION_REPAIR,
                    "rationale": "Нужен узкий ремонт конкретной строки.",
                    "confidence": "high",
                    "guidance": "Проверь буквальную связь владельца и услуги.",
                    "target_answer_ids": [41, 999],
                    "invalidate_artifact_keys": ["annotations_999"],
                    "acceptance_checks": ["raw_corpus_unchanged"],
                },
                allowed_actions={ACTION_TARGETED_ANNOTATION_REPAIR},
                permitted_answer_ids={41},
                permitted_artifact_keys={"annotations_41"},
                prior_decisions=[],
                incident_fingerprint="abc",
            )

    def test_same_action_cannot_loop_on_same_fingerprint(self) -> None:
        with self.assertRaises(OrchestratorContractError):
            validate_recovery_decision(
                {
                    "action": ACTION_DETERMINISTIC_FALLBACK,
                    "rationale": "Локальные попытки уже исчерпаны безопасно.",
                    "confidence": "high",
                    "guidance": "",
                    "target_answer_ids": [],
                    "invalidate_artifact_keys": [],
                    "acceptance_checks": [
                        "prompt_contract_valid",
                        "semantic_review_passed",
                    ],
                },
                allowed_actions={ACTION_DETERMINISTIC_FALLBACK},
                permitted_answer_ids=set(),
                permitted_artifact_keys=set(),
                prior_decisions=[
                    {
                        "incident_fingerprint": "abc",
                        "facts_digest": "facts-a",
                        "status": "failed",
                        "action": ACTION_DETERMINISTIC_FALLBACK,
                    }
                ],
                incident_fingerprint="abc",
                incident_facts_digest="facts-a",
            )

    def test_success_or_changed_facts_do_not_trigger_loop_guard(self) -> None:
        decision = {
            "action": ACTION_DETERMINISTIC_FALLBACK,
            "rationale": "Локальные попытки исчерпаны без безопасного результата.",
            "confidence": "high",
            "guidance": "",
            "target_answer_ids": [],
            "invalidate_artifact_keys": [],
            "acceptance_checks": [
                "prompt_contract_valid",
                "semantic_review_passed",
            ],
        }
        for prior in (
            {
                "incident_fingerprint": "abc",
                "facts_digest": "facts-a",
                "status": "succeeded",
                "action": ACTION_DETERMINISTIC_FALLBACK,
            },
            {
                "incident_fingerprint": "abc",
                "facts_digest": "facts-old",
                "status": "failed",
                "action": ACTION_DETERMINISTIC_FALLBACK,
            },
        ):
            with self.subTest(prior=prior):
                accepted = validate_recovery_decision(
                    decision,
                    allowed_actions={ACTION_DETERMINISTIC_FALLBACK},
                    permitted_answer_ids=set(),
                    permitted_artifact_keys=set(),
                    prior_decisions=[prior],
                    incident_fingerprint="abc",
                    incident_facts_digest="facts-a",
                )
                self.assertEqual(
                    accepted["action"],
                    ACTION_DETERMINISTIC_FALLBACK,
                )

    def test_local_validator_rejects_bool_nonpositive_and_oversized_scopes(self) -> None:
        base = {
            "action": ACTION_TARGETED_ANNOTATION_REPAIR,
            "rationale": "Нужен узкий ремонт конкретной строки ответа модели.",
            "confidence": "high",
            "guidance": "Проверь буквальную связь владельца и услуги.",
            "target_answer_ids": [1],
            "invalidate_artifact_keys": [],
            "acceptance_checks": ["raw_corpus_unchanged"],
        }
        invalid_cases = (
            {"target_answer_ids": [True]},
            {"target_answer_ids": [0]},
            {"target_answer_ids": list(range(1, 42))},
            {"invalidate_artifact_keys": [f"artifact-{i}" for i in range(21)]},
            {"acceptance_checks": ["raw_corpus_unchanged"] * 9},
            {"confidence": "low"},
        )
        for changes in invalid_cases:
            with self.subTest(changes=changes):
                with self.assertRaises(OrchestratorContractError):
                    validate_recovery_decision(
                        {**base, **changes},
                        allowed_actions={ACTION_TARGETED_ANNOTATION_REPAIR},
                        permitted_answer_ids=set(range(0, 50)),
                        permitted_artifact_keys={
                            f"artifact-{i}" for i in range(30)
                        },
                        prior_decisions=[],
                        incident_fingerprint="abc",
                        incident_facts_digest="facts-a",
                    )


class RecoveryPlannerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._enabled = patch(
            "app.services.recovery_orchestrator.settings."
            "PIPELINE_ORCHESTRATOR_ENABLED",
            True,
        )
        self._enabled.start()

    def tearDown(self) -> None:
        self._enabled.stop()

    async def test_disabled_orchestrator_fails_before_provider_call(self) -> None:
        with (
            patch(
                "app.services.recovery_orchestrator.settings."
                "PIPELINE_ORCHESTRATOR_ENABLED",
                False,
            ),
            patch(
                "app.services.recovery_orchestrator.chat",
                new_callable=AsyncMock,
            ) as chat_mock,
        ):
            with self.assertRaisesRegex(
                OrchestratorContractError,
                "orchestrator is disabled",
            ):
                await plan_recovery(
                    incident={"stage": "scenario_design"},
                    allowed_actions={ACTION_DETERMINISTIC_FALLBACK},
                )
        chat_mock.assert_not_awaited()

    async def test_fable_only_selects_from_stage_allowlist(self) -> None:
        parsed = {
            "action": ACTION_RETRY_WITH_GUIDANCE,
            "rationale": (
                "Последний кандидат близок к контракту; нужен один узкий ремонт."
            ),
            "confidence": "medium",
            "guidance": "Считай открытый вопрос измерением, а не утверждением.",
            "target_answer_ids": [],
            "invalidate_artifact_keys": [],
            "acceptance_checks": [
                "prompt_contract_valid",
                "semantic_review_passed",
            ],
        }
        with patch(
            "app.services.recovery_orchestrator.chat",
            new=AsyncMock(return_value=_chat_result(parsed)),
        ) as chat_mock:
            result = await plan_recovery(
                incident={
                    "stage": "scenario_design",
                    "code": "prompt_set_non_convergent",
                    "fingerprint": "incident-1",
                },
                allowed_actions={
                    ACTION_RETRY_WITH_GUIDANCE,
                    ACTION_DETERMINISTIC_FALLBACK,
                },
            )
        self.assertEqual(result.decision["action"], ACTION_RETRY_WITH_GUIDANCE)
        kwargs = chat_mock.await_args.kwargs
        self.assertEqual(kwargs["model"], "anthropic/claude-fable-5")
        self.assertEqual(kwargs["reasoning_effort"], "high")
        self.assertLessEqual(kwargs["max_tokens"], 4_000)
        self.assertFalse(kwargs["retry_response_contract_errors"])
        self.assertFalse(kwargs["retry_transport_errors"])

    async def test_invalid_model_scope_is_rejected_after_the_call(self) -> None:
        parsed = {
            "action": ACTION_TARGETED_ANNOTATION_REPAIR,
            "rationale": "Следует проверить только проблемный raw-ответ.",
            "confidence": "high",
            "guidance": "Скопируй буквальный непрерывный фрагмент.",
            "target_answer_ids": [78],
            "invalidate_artifact_keys": [],
            "acceptance_checks": ["raw_corpus_unchanged"],
        }
        with patch(
            "app.services.recovery_orchestrator.chat",
            new=AsyncMock(return_value=_chat_result(parsed)),
        ):
            with self.assertRaises(OrchestratorContractError):
                await plan_recovery(
                    incident={"stage": "knowledge_gap", "fingerprint": "x"},
                    allowed_actions={ACTION_TARGETED_ANNOTATION_REPAIR},
                    permitted_answer_ids={77},
                )

    async def test_history_payload_keeps_facts_digest_and_status(self) -> None:
        parsed = {
            "action": ACTION_DETERMINISTIC_FALLBACK,
            "rationale": "Нужен безопасный детерминированный резервный сценарий.",
            "confidence": "high",
            "guidance": "",
            "target_answer_ids": [],
            "invalidate_artifact_keys": [],
            "acceptance_checks": ["prompt_contract_valid"],
        }
        prior = {
            "incident_fingerprint": "old-fingerprint",
            "facts_digest": "old-facts",
            "status": "failed",
            "action": ACTION_RETRY_WITH_GUIDANCE,
        }
        with patch(
            "app.services.recovery_orchestrator.chat",
            new=AsyncMock(return_value=_chat_result(parsed)),
        ) as chat_mock:
            await plan_recovery(
                incident={
                    "stage": "scenario_design",
                    "fingerprint": "current-fingerprint",
                    "facts_digest": "current-facts",
                },
                allowed_actions={ACTION_DETERMINISTIC_FALLBACK},
                prior_decisions=[prior],
            )

        payload = json.loads(
            chat_mock.await_args.kwargs["messages"][1]["content"]
        )
        self.assertEqual(
            payload["prior_decisions"][0]["facts_digest"],
            "old-facts",
        )
        self.assertEqual(payload["prior_decisions"][0]["status"], "failed")


class DurableRecoveryStateTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        db_path = Path(self._temp_dir.name) / "recovery.sqlite3"
        self.engine = create_async_engine(
            f"sqlite+aiosqlite:///{db_path}",
            echo=False,
        )
        self.SessionLocal = async_sessionmaker(
            self.engine,
            expire_on_commit=False,
            class_=AsyncSession,
        )
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        self._session_patch = patch(
            "app.services.recovery_state.SessionLocal",
            self.SessionLocal,
        )
        self._session_patch.start()
        self._lease_session_patch = patch(
            "app.services.run_lease.SessionLocal",
            self.SessionLocal,
        )
        self._lease_session_patch.start()
        self._enabled_patch = patch(
            "app.services.recovery_state.settings."
            "PIPELINE_ORCHESTRATOR_ENABLED",
            True,
        )
        self._enabled_patch.start()
        self.run_id = str(uuid.uuid4())
        async with self.SessionLocal() as session:
            session.add(
                Run(
                    id=self.run_id,
                    domain="example.com",
                    status=RunStatus.analyzing,
                    config_json={},
                    attempt_count=1,
                    resume_count=0,
                )
            )
            await session.commit()

    async def asyncTearDown(self) -> None:
        self._enabled_patch.stop()
        self._lease_session_patch.stop()
        self._session_patch.stop()
        await self.engine.dispose()
        self._temp_dir.cleanup()

    async def test_disabled_state_does_not_reserve_or_spend_epoch(self) -> None:
        with (
            patch(
                "app.services.recovery_state.settings."
                "PIPELINE_ORCHESTRATOR_ENABLED",
                False,
            ),
            patch(
                "app.services.recovery_state.plan_recovery",
                new_callable=AsyncMock,
            ) as planner,
        ):
            with self.assertRaisesRegex(
                OrchestratorContractError,
                "orchestrator is disabled",
            ):
                await plan_durable_recovery(
                    self.run_id,
                    stage_key="scenario_design",
                    failure_class="repairable_semantic",
                    failure_code="prompt_set_non_convergent",
                    diagnostics={"validation_errors": ["x"]},
                    facts={"profile_sha256": "a" * 64},
                    allowed_actions={ACTION_DETERMINISTIC_FALLBACK},
                )
        planner.assert_not_awaited()
        async with self.SessionLocal() as session:
            count = len(
                list(
                    (
                        await session.execute(
                            select(RecoveryEpoch).where(
                                RecoveryEpoch.run_id == self.run_id
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
            )
        self.assertEqual(count, 0)

    async def test_unstarted_diagnosis_is_reused_without_spending_budget(self) -> None:
        diagnostics = {"validation_errors": ["x"]}
        facts = {"profile_sha256": "a" * 64}
        fingerprint = recovery_failure_fingerprint(
            stage_key="scenario_design",
            failure_class="repairable_semantic",
            failure_code="prompt_set_non_convergent",
            diagnostics=diagnostics,
        )
        facts_digest = recovery_scope_digest(
            facts=facts,
            allowed_actions={ACTION_DETERMINISTIC_FALLBACK},
        )
        async with self.SessionLocal() as session:
            prepared = RecoveryEpoch(
                run_id=self.run_id,
                epoch=1,
                stage_key="scenario_design",
                failure_class="repairable_semantic",
                failure_code="prompt_set_non_convergent",
                failure_fingerprint=fingerprint,
                facts_digest=facts_digest,
                status="diagnosing",
                # Legacy/pre-reservation rows may already carry the configured
                # model name; status, not model presence, proves call start.
                model=ORCHESTRATOR_MODEL,
                input_json={"planner_attempt": {"started": False}},
            )
            session.add(prepared)
            await session.commit()
            await session.refresh(prepared)
            prepared_id = prepared.id

        decision = {
            "action": ACTION_DETERMINISTIC_FALLBACK,
            "rationale": "Локальные попытки исчерпаны без безопасного результата.",
            "confidence": "high",
            "guidance": "",
            "target_answer_ids": [],
            "invalidate_artifact_keys": [],
            "acceptance_checks": ["prompt_contract_valid"],
            "incident_fingerprint": fingerprint,
            "orchestrator_version": ORCHESTRATOR_VERSION,
        }
        planned = OrchestratorResult(
            decision=decision,
            raw_text="{}",
            usage={"prompt_tokens": 100},
            input_digest="input-digest",
        )
        with (
            patch(
                "app.services.recovery_state.settings."
                "PIPELINE_ORCHESTRATOR_MAX_CALLS_PER_RUN",
                1,
            ),
            patch(
                "app.services.recovery_state.plan_recovery",
                new=AsyncMock(return_value=planned),
            ) as planner,
        ):
            result = await plan_durable_recovery(
                self.run_id,
                stage_key="scenario_design",
                failure_class="repairable_semantic",
                failure_code="prompt_set_non_convergent",
                diagnostics=diagnostics,
                facts=facts,
                allowed_actions={ACTION_DETERMINISTIC_FALLBACK},
            )

        self.assertEqual(result.epoch_id, prepared_id)
        self.assertFalse(result.reused)
        planner.assert_awaited_once()
        async with self.SessionLocal() as session:
            rows = list(
                (
                    await session.execute(
                        select(RecoveryEpoch).where(
                            RecoveryEpoch.run_id == self.run_id
                        )
                    )
                )
                .scalars()
                .all()
            )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].status, "planned")
        self.assertEqual(rows[0].model, ORCHESTRATOR_MODEL)
        self.assertTrue(rows[0].input_json["planner_attempt"]["started"])
        self.assertTrue(rows[0].input_json["planner_attempt"]["completed"])
        self.assertEqual(
            rows[0].usage_json["_aiv_orchestrator"]["input_digest"],
            "input-digest",
        )
        self.assertEqual(
            rows[0].usage_json["_aiv_orchestrator"]["raw_text"],
            "{}",
        )

    async def test_interrupted_planner_attempt_is_accounted_and_reconciled(self) -> None:
        diagnostics = {"validation_errors": ["x"]}
        facts = {"profile_sha256": "a" * 64}
        fingerprint = recovery_failure_fingerprint(
            stage_key="scenario_design",
            failure_class="repairable_semantic",
            failure_code="prompt_set_non_convergent",
            diagnostics=diagnostics,
        )
        facts_digest = recovery_scope_digest(
            facts=facts,
            allowed_actions={ACTION_DETERMINISTIC_FALLBACK},
        )
        async with self.SessionLocal() as session:
            session.add(
                RecoveryEpoch(
                    run_id=self.run_id,
                    epoch=1,
                    stage_key="scenario_design",
                    failure_class="repairable_semantic",
                    failure_code="prompt_set_non_convergent",
                    failure_fingerprint=fingerprint,
                    facts_digest=facts_digest,
                    status="planning",
                    model=ORCHESTRATOR_MODEL,
                    input_json={
                        "planner_attempt": {
                            "started": True,
                            "completed": False,
                        }
                    },
                )
            )
            await session.commit()

        decision = {
            "action": ACTION_DETERMINISTIC_FALLBACK,
            "rationale": "Локальные попытки исчерпаны без безопасного результата.",
            "confidence": "high",
            "guidance": "",
            "target_answer_ids": [],
            "invalidate_artifact_keys": [],
            "acceptance_checks": ["prompt_contract_valid"],
            "incident_fingerprint": fingerprint,
            "orchestrator_version": ORCHESTRATOR_VERSION,
        }
        planned = OrchestratorResult(
            decision=decision,
            raw_text="{}",
            usage={"prompt_tokens": 100},
            input_digest="input-digest",
        )
        with (
            patch(
                "app.services.recovery_state.settings."
                "PIPELINE_ORCHESTRATOR_MAX_CALLS_PER_RUN",
                2,
            ),
            patch(
                "app.services.recovery_state.plan_recovery",
                new=AsyncMock(return_value=planned),
            ) as planner,
        ):
            result = await plan_durable_recovery(
                self.run_id,
                stage_key="scenario_design",
                failure_class="repairable_semantic",
                failure_code="prompt_set_non_convergent",
                diagnostics=diagnostics,
                facts=facts,
                allowed_actions={ACTION_DETERMINISTIC_FALLBACK},
            )
            reused = await plan_durable_recovery(
                self.run_id,
                stage_key="scenario_design",
                failure_class="repairable_semantic",
                failure_code="prompt_set_non_convergent",
                diagnostics=diagnostics,
                facts=facts,
                allowed_actions={ACTION_DETERMINISTIC_FALLBACK},
            )

        planner.assert_awaited_once()
        self.assertTrue(reused.reused)
        self.assertEqual(reused.epoch_id, result.epoch_id)
        async with self.SessionLocal() as session:
            rows = list(
                (
                    await session.execute(
                        select(RecoveryEpoch)
                        .where(RecoveryEpoch.run_id == self.run_id)
                        .order_by(RecoveryEpoch.epoch)
                    )
                )
                .scalars()
                .all()
            )
        self.assertEqual([row.status for row in rows], ["failed", "planned"])
        self.assertTrue(
            rows[0].usage_json["planner_attempt"]["reconciled_after_restart"]
        )

    async def test_stage_planner_budget_blocks_second_post_after_restart(
        self,
    ) -> None:
        diagnostics = {"critic_iteration": 2, "issue_answer_ids": [569]}
        facts = {"critic_review_sha256": "a" * 64}
        fingerprint = recovery_failure_fingerprint(
            stage_key="analysis_critic",
            failure_class="repairable_semantic",
            failure_code="analysis_critic_non_convergent",
            diagnostics=diagnostics,
        )
        facts_digest = recovery_scope_digest(
            facts=facts,
            allowed_actions={ACTION_TARGETED_ANNOTATION_REPAIR},
            permitted_answer_ids={569},
        )
        async with self.SessionLocal() as session:
            session.add(
                RecoveryEpoch(
                    run_id=self.run_id,
                    epoch=1,
                    stage_key="analysis_critic",
                    failure_class="repairable_semantic",
                    failure_code="analysis_critic_non_convergent",
                    failure_fingerprint=fingerprint,
                    facts_digest=facts_digest,
                    status="planning",
                    model=ORCHESTRATOR_MODEL,
                    input_json={
                        "planner_attempt": {
                            "started": True,
                            "completed": False,
                        }
                    },
                )
            )
            await session.commit()

        with patch(
            "app.services.recovery_state.plan_recovery",
            new_callable=AsyncMock,
        ) as planner:
            with self.assertRaisesRegex(
                RecoveryBudgetExceeded,
                "stage planner call budget exhausted",
            ):
                await plan_durable_recovery(
                    self.run_id,
                    stage_key="analysis_critic",
                    failure_class="repairable_semantic",
                    failure_code="analysis_critic_non_convergent",
                    diagnostics=diagnostics,
                    facts=facts,
                    allowed_actions={ACTION_TARGETED_ANNOTATION_REPAIR},
                    permitted_answer_ids={569},
                    stage_planner_call_limit=1,
                )
        planner.assert_not_awaited()
        async with self.SessionLocal() as session:
            epoch = (
                await session.execute(
                    select(RecoveryEpoch).where(
                        RecoveryEpoch.run_id == self.run_id
                    )
                )
            ).scalar_one()
        self.assertEqual(epoch.status, "failed")
        self.assertTrue(
            epoch.usage_json["planner_attempt"][
                "reconciled_after_restart"
            ]
        )

    async def test_plan_survives_restart_and_finishes_once(self) -> None:
        decision = {
            "action": ACTION_DETERMINISTIC_FALLBACK,
            "rationale": "Локальные попытки исчерпаны без безопасного результата.",
            "confidence": "high",
            "guidance": "",
            "target_answer_ids": [],
            "invalidate_artifact_keys": [],
            "acceptance_checks": [
                "prompt_contract_valid",
                "semantic_review_passed",
            ],
            "incident_fingerprint": "planner-fingerprint",
            "orchestrator_version": "aiv-recovery-orchestrator-v1",
        }
        planned = OrchestratorResult(
            decision=decision,
            raw_text="{}",
            usage={"prompt_tokens": 100},
            input_digest="input-digest",
        )
        kwargs = {
            "stage_key": "scenario_design",
            "failure_class": "repairable_semantic",
            "failure_code": "prompt_set_non_convergent",
            "diagnostics": {"validation_errors": ["x"]},
            "facts": {"profile_sha256": "a" * 64},
            "allowed_actions": {ACTION_DETERMINISTIC_FALLBACK},
        }
        with patch(
            "app.services.recovery_state.plan_recovery",
            new=AsyncMock(return_value=planned),
        ) as planner:
            first = await plan_durable_recovery(self.run_id, **kwargs)
            second = await plan_durable_recovery(self.run_id, **kwargs)

        self.assertFalse(first.reused)
        self.assertTrue(second.reused)
        self.assertEqual(first.run_id, self.run_id)
        self.assertEqual(second.run_id, self.run_id)
        self.assertEqual(first.epoch_id, second.epoch_id)
        planner.assert_awaited_once()

        await mark_recovery_executing(second)
        await finish_recovery(
            second,
            succeeded=True,
            before_digest="before",
            after_digest="after",
            details={"accepted": True},
        )
        async with self.SessionLocal() as session:
            epochs = list(
                (
                    await session.execute(
                        select(RecoveryEpoch).where(
                            RecoveryEpoch.run_id == self.run_id
                        )
                    )
                )
                .scalars()
                .all()
            )
        self.assertEqual(len(epochs), 1)
        self.assertEqual(epochs[0].status, "succeeded")
        self.assertEqual(epochs[0].plan_json, decision)
        self.assertTrue(epochs[0].outcome_json["succeeded"])
        self.assertEqual(epochs[0].outcome_json["execution_attempts"], 1)

    async def test_reused_plan_is_scope_validated_and_digest_checked(self) -> None:
        decision = {
            "action": ACTION_DETERMINISTIC_FALLBACK,
            "rationale": "Локальные попытки исчерпаны без безопасного результата.",
            "confidence": "high",
            "guidance": "",
            "target_answer_ids": [],
            "invalidate_artifact_keys": [],
            "acceptance_checks": [
                "prompt_contract_valid",
                "semantic_review_passed",
            ],
            "incident_fingerprint": "planner-fingerprint",
            "orchestrator_version": ORCHESTRATOR_VERSION,
        }
        planned = OrchestratorResult(
            decision=decision,
            raw_text=json.dumps(decision, ensure_ascii=False),
            usage={"prompt_tokens": 100},
            input_digest="input-digest",
        )
        kwargs = {
            "stage_key": "scenario_design",
            "failure_class": "repairable_semantic",
            "failure_code": "prompt_set_non_convergent",
            "diagnostics": {"validation_errors": ["x"]},
            "facts": {"profile_sha256": "a" * 64},
            "allowed_actions": {ACTION_DETERMINISTIC_FALLBACK},
        }
        with patch(
            "app.services.recovery_state.plan_recovery",
            new=AsyncMock(return_value=planned),
        ) as planner:
            first = await plan_durable_recovery(self.run_id, **kwargs)

            out_of_scope = {
                **decision,
                "action": ACTION_TARGETED_ANNOTATION_REPAIR,
                "target_answer_ids": [999],
                "acceptance_checks": ["raw_corpus_unchanged"],
            }
            async with self.SessionLocal() as session:
                await session.execute(
                    update(RecoveryEpoch)
                    .where(RecoveryEpoch.id == first.epoch_id)
                    .values(
                        plan_json=out_of_scope,
                        plan_digest=stable_digest(out_of_scope),
                    )
                )
                await session.commit()
            with self.assertRaisesRegex(
                OrchestratorContractError,
                "not allowed|outside the incident",
            ):
                await plan_durable_recovery(self.run_id, **kwargs)

            async with self.SessionLocal() as session:
                await session.execute(
                    update(RecoveryEpoch)
                    .where(RecoveryEpoch.id == first.epoch_id)
                    .values(
                        plan_json=decision,
                        plan_digest=stable_digest(decision),
                    )
                )
                await session.commit()
            reused = await plan_durable_recovery(self.run_id, **kwargs)
            self.assertTrue(reused.reused)

            async with self.SessionLocal() as session:
                await session.execute(
                    update(RecoveryEpoch)
                    .where(RecoveryEpoch.id == first.epoch_id)
                    .values(plan_digest="0" * 64)
                )
                await session.commit()
            with self.assertRaisesRegex(
                OrchestratorContractError,
                "digest mismatch",
            ):
                await mark_recovery_executing(reused)

        planner.assert_awaited_once()

    async def test_lost_lease_after_planner_cannot_publish_plan(self) -> None:
        decision = {
            "action": ACTION_DETERMINISTIC_FALLBACK,
            "rationale": "Локальные попытки исчерпаны без безопасного результата.",
            "confidence": "high",
            "guidance": "",
            "target_answer_ids": [],
            "invalidate_artifact_keys": [],
            "acceptance_checks": [
                "prompt_contract_valid",
                "semantic_review_passed",
            ],
            "incident_fingerprint": "planner-fingerprint",
            "orchestrator_version": "aiv-recovery-orchestrator-v1",
        }
        planned = OrchestratorResult(
            decision=decision,
            raw_text="{}",
            usage={"prompt_tokens": 100},
            input_digest="input-digest",
        )
        async with self.SessionLocal() as session:
            await session.execute(
                update(Run)
                .where(Run.id == self.run_id)
                .values(execution_slot=1, lease_owner="current-owner")
            )
            await session.commit()

        async def lose_lease_during_planning(**_kwargs):
            async with self.SessionLocal() as session:
                await session.execute(
                    update(Run)
                    .where(Run.id == self.run_id)
                    .values(lease_owner="replacement-owner")
                )
                await session.commit()
            return planned

        with patch(
            "app.services.recovery_state.plan_recovery",
            new=AsyncMock(side_effect=lose_lease_during_planning),
        ):
            with bind_run_lease(self.run_id, "current-owner"):
                with self.assertRaises(RunLeaseLostError):
                    await plan_durable_recovery(
                        self.run_id,
                        stage_key="scenario_design",
                        failure_class="repairable_semantic",
                        failure_code="prompt_set_non_convergent",
                        diagnostics={"validation_errors": ["x"]},
                        facts={"profile_sha256": "a" * 64},
                        allowed_actions={ACTION_DETERMINISTIC_FALLBACK},
                    )

        async with self.SessionLocal() as session:
            epoch = (
                await session.execute(
                    select(RecoveryEpoch).where(
                        RecoveryEpoch.run_id == self.run_id
                    )
                )
            ).scalar_one()
        self.assertEqual(epoch.status, "planning")
        self.assertEqual(epoch.model, ORCHESTRATOR_MODEL)
        self.assertTrue(epoch.input_json["planner_attempt"]["started"])
        self.assertFalse(epoch.input_json["planner_attempt"]["completed"])
        self.assertIsNone(epoch.plan_json)

    async def test_stale_lease_cannot_mark_or_finish_recovery(self) -> None:
        decision = {
            "action": ACTION_DETERMINISTIC_FALLBACK,
            "rationale": "Локальные попытки исчерпаны без безопасного результата.",
            "confidence": "high",
            "guidance": "",
            "target_answer_ids": [],
            "invalidate_artifact_keys": [],
            "acceptance_checks": [
                "prompt_contract_valid",
                "semantic_review_passed",
            ],
            "incident_fingerprint": "planner-fingerprint",
            "orchestrator_version": "aiv-recovery-orchestrator-v1",
        }
        planned = OrchestratorResult(
            decision=decision,
            raw_text="{}",
            usage={"prompt_tokens": 100},
            input_digest="input-digest",
        )
        with patch(
            "app.services.recovery_state.plan_recovery",
            new=AsyncMock(return_value=planned),
        ):
            plan = await plan_durable_recovery(
                self.run_id,
                stage_key="scenario_design",
                failure_class="repairable_semantic",
                failure_code="prompt_set_non_convergent",
                diagnostics={"validation_errors": ["x"]},
                facts={"profile_sha256": "a" * 64},
                allowed_actions={ACTION_DETERMINISTIC_FALLBACK},
            )

        async with self.SessionLocal() as session:
            await session.execute(
                update(Run)
                .where(Run.id == self.run_id)
                .values(execution_slot=1, lease_owner="current-owner")
            )
            await session.commit()

        with bind_run_lease(self.run_id, "stale-owner"):
            with self.assertRaises(RunLeaseLostError):
                await mark_recovery_executing(plan)
        async with self.SessionLocal() as session:
            status = (
                await session.execute(
                    select(RecoveryEpoch.status).where(
                        RecoveryEpoch.id == plan.epoch_id
                    )
                )
            ).scalar_one()
        self.assertEqual(status, "planned")

        with bind_run_lease(self.run_id, "current-owner"):
            await mark_recovery_executing(plan)
        async with self.SessionLocal() as session:
            await session.execute(
                update(Run)
                .where(Run.id == self.run_id)
                .values(lease_owner="replacement-owner")
            )
            await session.commit()

        with bind_run_lease(self.run_id, "current-owner"):
            with self.assertRaises(RunLeaseLostError):
                await finish_recovery(
                    plan,
                    succeeded=True,
                    before_digest="before",
                    after_digest="after",
                )
        async with self.SessionLocal() as session:
            epoch = (
                await session.execute(
                    select(RecoveryEpoch).where(
                        RecoveryEpoch.id == plan.epoch_id
                    )
                )
            ).scalar_one()
        self.assertEqual(epoch.status, "executing")
        self.assertEqual(epoch.outcome_json["execution_attempts"], 1)

    async def test_execution_attempt_budget_survives_worker_restarts(self) -> None:
        decision = {
            "action": ACTION_DETERMINISTIC_FALLBACK,
            "rationale": "Локальные попытки исчерпаны без безопасного результата.",
            "confidence": "high",
            "guidance": "",
            "target_answer_ids": [],
            "invalidate_artifact_keys": [],
            "acceptance_checks": [
                "prompt_contract_valid",
                "semantic_review_passed",
            ],
            "incident_fingerprint": "planner-fingerprint",
            "orchestrator_version": "aiv-recovery-orchestrator-v1",
        }
        planned = OrchestratorResult(
            decision=decision,
            raw_text="{}",
            usage={"prompt_tokens": 100},
            input_digest="input-digest",
        )
        kwargs = {
            "stage_key": "scenario_design",
            "failure_class": "repairable_semantic",
            "failure_code": "prompt_set_non_convergent",
            "diagnostics": {"validation_errors": ["x"]},
            "facts": {"profile_sha256": "a" * 64},
            "allowed_actions": {ACTION_DETERMINISTIC_FALLBACK},
        }
        with patch(
            "app.services.recovery_state.plan_recovery",
            new=AsyncMock(return_value=planned),
        ) as planner:
            first_worker = await plan_durable_recovery(self.run_id, **kwargs)
            await mark_recovery_executing(first_worker)
            second_worker = await plan_durable_recovery(self.run_id, **kwargs)
            await mark_recovery_executing(second_worker)
            third_worker = await plan_durable_recovery(self.run_id, **kwargs)
            with self.assertRaises(RecoveryBudgetExceeded):
                await mark_recovery_executing(third_worker)

        planner.assert_awaited_once()
        self.assertTrue(second_worker.reused)
        self.assertTrue(third_worker.reused)
        async with self.SessionLocal() as session:
            epoch = (
                await session.execute(
                    select(RecoveryEpoch).where(
                        RecoveryEpoch.id == first_worker.epoch_id
                    )
                )
            ).scalar_one()
        self.assertEqual(epoch.status, "blocked")
        self.assertEqual(epoch.outcome_json["execution_attempts"], 2)
        self.assertEqual(
            epoch.outcome_json["reason"],
            "execution_attempt_budget_exhausted",
        )


if __name__ == "__main__":
    unittest.main()
