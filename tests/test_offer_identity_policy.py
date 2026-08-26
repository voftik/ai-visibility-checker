from __future__ import annotations

import copy
import unittest
from dataclasses import replace

from app.services.offer_catalog import (
    OfferCandidate,
    OfferKind,
    SourceUnit,
    build_offer_catalog,
)
from app.services.offer_identity_policy import (
    OFFER_IDENTITY_MODEL_BATCH_VERSION,
    OFFER_IDENTITY_MODEL_DECISION_VERSION,
    OFFER_IDENTITY_POLICY_DIGEST,
    OfferIdentityContract,
    OfferIdentityDecision,
    OfferIdentityModelRole,
    OfferIdentityPolicyError,
    OfferIdentityPolicyResult,
    OfferIdentityReasonCode,
    build_offer_identity_contract,
    build_offer_identity_model_request,
    offer_identity_decision_schema,
    resolve_offer_identity_policy,
)


def _catalog():
    definitions = (
        (
            "Garpun",
            (),
            OfferKind.PRODUCT,
            "Example развивает продукт Garpun для управления рекламой.",
        ),
        (
            "Centra",
            (),
            OfferKind.PRODUCT,
            "Example предлагает платформу Centra для покупки рекламы.",
        ),
        (
            "Campaign 360",
            ("campaign",),
            OfferKind.PRODUCT,
            (
                "Example предлагает продукт Campaign 360, также называемый "
                "campaign, для медиапланирования."
            ),
        ),
        (
            "Orbit Cloud",
            (),
            OfferKind.PRODUCT,
            "Example развивает платформу Orbit Cloud для аналитики.",
        ),
        (
            "Running shoes",
            (),
            OfferKind.PRODUCT,
            "Example предлагает продукт Running shoes для спортсменов.",
        ),
        (
            "eCommerce",
            (),
            OfferKind.PRODUCT,
            "Example предлагает направление eCommerce для магазинов.",
        ),
        (
            "iGaming",
            (),
            OfferKind.PRODUCT,
            "Example предлагает решение iGaming для операторов.",
        ),
        (
            "SEO",
            (),
            OfferKind.SERVICE,
            "Example предлагает услугу SEO для интернет-магазинов.",
        ),
    )
    sources = []
    candidates = []
    for index, (name, aliases, kind, text) in enumerate(definitions):
        source = SourceUnit.from_text(
            source_unit_id=f"page:{index}",
            source_url=f"https://example.com/offers/{index}",
            text=text,
        )
        sources.append(source)
        candidates.append(
            OfferCandidate(
                canonical_name=name,
                aliases=aliases,
                kind=kind,
                source_url=source.source_url,
                evidence_excerpt=text,
                source_unit_id=source.source_unit_id,
                source_sha256=source.source_sha256,
                confidence=0.95,
                user_jobs=(),
            )
        )
    return build_offer_catalog(
        client_domain="example.com",
        client_aliases=("Example",),
        source_units=tuple(sources),
        candidates=tuple(candidates),
    )


def _reason_for(decision: OfferIdentityDecision) -> OfferIdentityReasonCode:
    if decision is OfferIdentityDecision.NAMED_OFFERING:
        return OfferIdentityReasonCode.EXPLICIT_NAMED_IDENTITY
    if decision is OfferIdentityDecision.GENERIC_CATEGORY:
        return OfferIdentityReasonCode.DESCRIPTIVE_OFFERING_PHRASE
    return OfferIdentityReasonCode.INSUFFICIENT_IDENTITY_EVIDENCE


def _model_row(
    contract,
    subject,
    *,
    role: OfferIdentityModelRole,
    decision: OfferIdentityDecision,
):
    return {
        "version": OFFER_IDENTITY_MODEL_DECISION_VERSION,
        "role": role.value,
        "subject_id": subject.subject_id,
        "input_digest": contract.input_digest,
        "policy_digest": contract.policy_digest,
        "catalog_digest": contract.catalog_digest,
        "subject_digest": subject.subject_digest,
        "evidence_refs_digest": subject.evidence_refs_digest,
        "reviewed_evidence_ref_digests": list(subject.evidence_ref_digests),
        "decision": decision.value,
        "reason_code": _reason_for(decision).value,
        "rationale": "Классификация опирается только на переданные источники.",
    }


def _model_batch(
    contract,
    *,
    role: OfferIdentityModelRole,
    decisions: dict[str, OfferIdentityDecision],
):
    request = build_offer_identity_model_request(contract, role=role)
    return {
        "version": OFFER_IDENTITY_MODEL_BATCH_VERSION,
        "role": role.value,
        "input_digest": contract.input_digest,
        "policy_digest": contract.policy_digest,
        "catalog_digest": contract.catalog_digest,
        "request_contract_digest": request["payload"]["request_contract"][
            "request_contract_digest"
        ],
        "subject_count": len(contract.subjects),
        "decisions": [
            _model_row(
                contract,
                subject,
                role=role,
                decision=decisions[subject.name],
            )
            for subject in contract.subjects
        ],
    }


def _decision_map():
    named = {"Garpun", "Centra", "Campaign 360", "Orbit Cloud"}
    return {
        "Garpun": OfferIdentityDecision.NAMED_OFFERING,
        "Centra": OfferIdentityDecision.NAMED_OFFERING,
        "Campaign 360": OfferIdentityDecision.NAMED_OFFERING,
        "Orbit Cloud": OfferIdentityDecision.NAMED_OFFERING,
        "Running shoes": OfferIdentityDecision.GENERIC_CATEGORY,
        "eCommerce": OfferIdentityDecision.GENERIC_CATEGORY,
        "iGaming": OfferIdentityDecision.GENERIC_CATEGORY,
        "SEO": OfferIdentityDecision.NAMED_OFFERING,
        "campaign": OfferIdentityDecision.GENERIC_CATEGORY,
    }, named


class OfferIdentityContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = _catalog()
        self.contract = build_offer_identity_contract(self.catalog)

    def test_enumerates_every_canonical_and_alias_exactly_once(self) -> None:
        expected_count = sum(
            1 + len(offer.aliases) for offer in self.catalog.accepted_offers
        )
        self.assertEqual(len(self.contract.subjects), expected_count)
        self.assertEqual(
            [item.ordinal for item in self.contract.subjects],
            list(range(expected_count)),
        )
        self.assertEqual(
            len({item.subject_id for item in self.contract.subjects}),
            expected_count,
        )
        observed = [
            (item.offer_id, item.name_role.value, item.name)
            for item in self.contract.subjects
        ]
        expected = [
            occurrence
            for offer in self.catalog.accepted_offers
            for occurrence in [
                (offer.offer_id, "canonical", offer.canonical_name),
                *(
                    (offer.offer_id, "alias", alias)
                    for alias in offer.aliases
                ),
            ]
        ]
        self.assertEqual(observed, expected)
        self.assertTrue(
            all(item.evidence_refs for item in self.contract.subjects)
        )

    def test_round_trip_is_content_addressed_and_catalog_bound(self) -> None:
        restored = OfferIdentityContract.from_mapping(
            self.contract.as_dict(), catalog=self.catalog
        )
        self.assertEqual(restored, self.contract)
        repeated = build_offer_identity_contract(self.catalog)
        self.assertEqual(repeated.input_digest, self.contract.input_digest)
        self.assertEqual(repeated.as_dict(), self.contract.as_dict())
        self.assertEqual(self.contract.policy_digest, OFFER_IDENTITY_POLICY_DIGEST)

    def test_tampered_evidence_is_rejected_even_with_recomputed_outer_digest(self) -> None:
        tampered = self.contract.as_dict()
        tampered["subjects"][0]["evidence_refs"][0]["evidence_excerpt"] += " X"
        # Recomputing the outer digest cannot repair the exact subject/catalog
        # binding, and callers should not need to trust a persisted mapping.
        with self.assertRaisesRegex(
            OfferIdentityPolicyError, "evidence ref digests mismatch"
        ):
            OfferIdentityContract.from_mapping(tampered, catalog=self.catalog)

    def test_missing_extra_and_duplicate_contract_subjects_are_rejected(self) -> None:
        missing = self.contract.as_dict()
        missing["subjects"].pop()
        missing["subject_count"] -= 1
        with self.assertRaises(OfferIdentityPolicyError):
            OfferIdentityContract.from_mapping(missing, catalog=self.catalog)

        extra = self.contract.as_dict()
        extra["unexpected"] = True
        with self.assertRaisesRegex(OfferIdentityPolicyError, "keys mismatch"):
            OfferIdentityContract.from_mapping(extra, catalog=self.catalog)

        duplicate = self.contract.as_dict()
        duplicate["subjects"][1] = copy.deepcopy(duplicate["subjects"][0])
        with self.assertRaises(OfferIdentityPolicyError):
            OfferIdentityContract.from_mapping(duplicate, catalog=self.catalog)

    def test_exactly_two_aggregate_requests_cover_all_subjects(self) -> None:
        primary = build_offer_identity_model_request(
            self.contract,
            role=OfferIdentityModelRole.PRIMARY,
        )
        critic = build_offer_identity_model_request(
            self.contract,
            role=OfferIdentityModelRole.CRITIC,
        )
        self.assertEqual(
            primary["payload"]["subjects"], critic["payload"]["subjects"]
        )
        self.assertNotIn("primary_result", critic["payload"])
        self.assertNotEqual(primary["request_digest"], critic["request_digest"])
        self.assertEqual(
            primary["response_schema"]["properties"]["role"]["const"],
            "primary",
        )
        self.assertEqual(
            critic["response_schema"]["properties"]["role"]["const"],
            "critic",
        )
        self.assertEqual(
            primary["payload"]["subjects"][0]["evidence_refs"],
            [item.as_dict() for item in self.catalog.accepted_offers[0].evidence_refs],
        )
        self.assertEqual(
            offer_identity_decision_schema(self.contract, "primary")[
                "additionalProperties"
            ],
            False,
        )
        schema = primary["response_schema"]
        self.assertEqual(
            schema["properties"]["decisions"]["minItems"],
            len(self.contract.subjects),
        )
        self.assertEqual(
            schema["properties"]["decisions"]["maxItems"],
            len(self.contract.subjects),
        )
        self.assertEqual(primary["model_parameters"]["temperature"], 0.15)
        self.assertEqual(
            primary["payload"]["request_contract"]["model_parameters"],
            primary["model_parameters"],
        )


class OfferIdentityResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = _catalog()
        self.contract = build_offer_identity_contract(self.catalog)
        decisions, self.expected_named = _decision_map()
        self.decisions = decisions

    def _batch(self, role: OfferIdentityModelRole):
        return _model_batch(
            self.contract, role=role, decisions=self.decisions
        )

    def test_exact_independent_agreement_is_the_only_standalone_path(self) -> None:
        result = resolve_offer_identity_policy(
            self.catalog,
            contract=self.contract,
            primary_results=self._batch(OfferIdentityModelRole.PRIMARY),
            critic_results=self._batch(OfferIdentityModelRole.CRITIC),
        )
        standalone = {item.name for item in result.decisions if item.standalone}
        self.assertEqual(standalone, self.expected_named)
        by_name = {item.name: item for item in result.decisions}
        for name in ("Running shoes", "eCommerce", "iGaming", "campaign"):
            self.assertEqual(
                by_name[name].decision, OfferIdentityDecision.GENERIC_CATEGORY
            )
            self.assertFalse(by_name[name].standalone)
        # Static vocabulary beats two erroneous model votes.
        self.assertEqual(
            by_name["SEO"].resolution_code, "static_generic_override"
        )
        self.assertEqual(
            by_name["SEO"].decision, OfferIdentityDecision.GENERIC_CATEGORY
        )
        self.assertFalse(by_name["SEO"].standalone)
        self.assertEqual(result.output_digest, result.as_dict()["output_digest"])

    def test_disagreement_maps_only_that_subject_to_ambiguous(self) -> None:
        primary = self._batch(OfferIdentityModelRole.PRIMARY)
        critic = self._batch(OfferIdentityModelRole.CRITIC)
        target = next(
            item for item in self.contract.subjects if item.name == "Garpun"
        )
        index = [item.subject_id for item in self.contract.subjects].index(
            target.subject_id
        )
        critic["decisions"][index] = _model_row(
            self.contract,
            target,
            role=OfferIdentityModelRole.CRITIC,
            decision=OfferIdentityDecision.GENERIC_CATEGORY,
        )
        result = resolve_offer_identity_policy(
            self.catalog,
            contract=self.contract,
            primary_results=primary,
            critic_results=critic,
        )
        by_name = {item.name: item for item in result.decisions}
        self.assertEqual(
            by_name["Garpun"].decision, OfferIdentityDecision.AMBIGUOUS
        )
        self.assertFalse(by_name["Garpun"].standalone)
        self.assertTrue(by_name["Centra"].standalone)

    def test_missing_invalid_and_duplicate_results_fail_closed_without_raise(self) -> None:
        primary = self._batch(OfferIdentityModelRole.PRIMARY)
        critic = self._batch(OfferIdentityModelRole.CRITIC)
        missing_subject = self.contract.subjects[0]
        primary["decisions"] = [
            item
            for item in primary["decisions"]
            if item["subject_id"] != missing_subject.subject_id
        ]
        duplicate_subject = self.contract.subjects[1]
        duplicate = next(
            item
            for item in critic["decisions"]
            if item["subject_id"] == duplicate_subject.subject_id
        )
        critic["decisions"].append(copy.deepcopy(duplicate))
        result = resolve_offer_identity_policy(
            self.catalog,
            contract=self.contract,
            primary_results=primary,
            critic_results=critic,
        )
        self.assertTrue(all(not item.standalone for item in result.decisions))
        self.assertIn("primary:role_invalid", result.diagnostics)
        self.assertIn("critic:role_invalid", result.diagnostics)

    def test_extra_unbound_result_invalidates_role_but_not_report(self) -> None:
        primary = self._batch(OfferIdentityModelRole.PRIMARY)
        extra = copy.deepcopy(primary["decisions"][0])
        extra["subject_id"] = "identity:" + "f" * 64
        primary["decisions"][-1] = extra
        result = resolve_offer_identity_policy(
            self.catalog,
            contract=self.contract,
            primary_results=primary,
            critic_results=self._batch(OfferIdentityModelRole.CRITIC),
        )
        self.assertTrue(
            all(
                not item.standalone
                for item in result.decisions
            )
        )
        self.assertIn("primary:role_invalid", result.diagnostics)

    def test_provider_error_and_stale_contract_rebuild_are_fail_closed(self) -> None:
        result = resolve_offer_identity_policy(
            self.catalog,
            contract=replace(self.contract, input_digest="f" * 64),
            primary_results=RuntimeError("provider unavailable"),
            critic_results=self._batch(OfferIdentityModelRole.CRITIC),
        )
        self.assertIn("contract_rebuilt_fail_closed", result.diagnostics)
        self.assertTrue(all(not item.standalone for item in result.decisions))
        by_name = {item.name: item for item in result.decisions}
        self.assertEqual(
            by_name["SEO"].decision, OfferIdentityDecision.GENERIC_CATEGORY
        )

    def test_repeated_resolution_has_identical_output_digest(self) -> None:
        kwargs = {
            "contract": self.contract,
            "primary_results": self._batch(OfferIdentityModelRole.PRIMARY),
            "critic_results": self._batch(OfferIdentityModelRole.CRITIC),
        }
        first = resolve_offer_identity_policy(self.catalog, **kwargs)
        second = resolve_offer_identity_policy(self.catalog, **kwargs)
        self.assertEqual(first.output_digest, second.output_digest)
        self.assertEqual(first.as_dict(), second.as_dict())

    def test_persisted_result_is_revalidated_against_current_catalog(self) -> None:
        result = resolve_offer_identity_policy(
            self.catalog,
            contract=self.contract,
            primary_results=self._batch(OfferIdentityModelRole.PRIMARY),
            critic_results=self._batch(OfferIdentityModelRole.CRITIC),
        )
        restored = OfferIdentityPolicyResult.from_mapping(
            result.as_dict(), catalog=self.catalog
        )
        self.assertEqual(restored, result)

        tampered = result.as_dict()
        tampered["decisions"][0]["standalone"] = not tampered["decisions"][0][
            "standalone"
        ]
        with self.assertRaises(OfferIdentityPolicyError):
            OfferIdentityPolicyResult.from_mapping(
                tampered, catalog=self.catalog
            )

        stale = result.as_dict()
        stale["input_digest"] = "f" * 64
        with self.assertRaisesRegex(OfferIdentityPolicyError, "input_digest"):
            OfferIdentityPolicyResult.from_mapping(stale, catalog=self.catalog)


if __name__ == "__main__":
    unittest.main()
