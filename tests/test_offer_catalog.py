from __future__ import annotations

from dataclasses import replace
import hashlib
import random
import unittest

from app.services.offer_catalog import (
    INTENT_CODES,
    MAX_ACCEPTED_OFFERS,
    AnswerSetReceipt,
    ClusterExclusion,
    IntentPrompt,
    OfferCandidate,
    OfferCatalog,
    OfferCatalogError,
    OfferEvidenceError,
    OfferKind,
    PromptCoverageError,
    PromptFoundation,
    ResumeCompatibilityError,
    SourceUnit,
    UpstreamArtifactDigests,
    artifact_digest,
    audit_resume_compatibility,
    build_answer_set_receipt,
    build_domain_research_payload,
    build_offer_catalog,
    build_offer_clusters,
    build_prompt_foundation,
    build_upstream_artifact_digests,
    normalize_offer_candidates_against_sources,
    reconstruct_domain_research_payload,
    validate_resume_compatibility,
)


def _candidate(
    source: SourceUnit,
    *,
    name: str,
    excerpt: str,
    aliases: tuple[str, ...] = (),
    kind: OfferKind = OfferKind.PRODUCT,
    confidence: float = 0.9,
    jobs: tuple[str, ...] = ("Выбрать решение",),
) -> OfferCandidate:
    return OfferCandidate(
        canonical_name=name,
        aliases=aliases,
        kind=kind,
        source_url=source.source_url,
        evidence_excerpt=excerpt,
        source_unit_id=source.source_unit_id,
        source_sha256=source.source_sha256,
        confidence=confidence,
        user_jobs=jobs,
    )


def _candidate_mapping(
    source: SourceUnit,
    *,
    name: str,
    excerpt: str,
) -> dict[str, object]:
    return {
        "canonical_name": name,
        "aliases": [],
        "kind": OfferKind.PRODUCT.value,
        "source_url": source.source_url,
        "evidence_excerpt": excerpt,
        "source_unit_id": source.source_unit_id,
        "source_sha256": source.source_sha256,
        "confidence": 0.9,
        "user_jobs": ["Выбрать решение"],
        "commercially_relevant": True,
    }


def _six_prompts(
    clusters: tuple,
    *,
    cover: bool = True,
) -> tuple[IntentPrompt, ...]:
    cluster_ids = tuple(cluster.cluster_id for cluster in clusters)
    return tuple(
        IntentPrompt(
            prompt_key=f"prompt:{intent.casefold()}",
            intent=intent,
            text=f"Пользовательский сценарий {intent}",
            supporting_cluster_ids=cluster_ids if cover and index == 0 else (),
        )
        for index, intent in enumerate(INTENT_CODES)
    )


class OfferCatalogEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = (
            "Realweb развивает платформу AdTech Compass для медиапланирования.\n"
            "Realweb предлагает услугу programmatic для крупных брендов.\n"
            "Запрос пользователя: DOOH и performance marketing. Сравните агентства."
        )
        self.source = SourceUnit.from_text(
            source_unit_id="page:home",
            source_url="https://realweb.ru/",
            text=self.text,
        )

    def test_accepts_specific_product_and_explicitly_bound_generic_service(self) -> None:
        product_excerpt = "Realweb развивает платформу AdTech Compass для медиапланирования."
        service_excerpt = "Realweb предлагает услугу programmatic для крупных брендов."

        catalog = build_offer_catalog(
            client_domain="realweb.ru",
            client_aliases=("Realweb", "Риалвеб"),
            source_units=(self.source,),
            candidates=(
                _candidate(
                    self.source,
                    name="AdTech Compass",
                    aliases=("Compass",),
                    excerpt=product_excerpt,
                    jobs=("Составить медиаплан",),
                ),
                _candidate(
                    self.source,
                    name="programmatic",
                    excerpt=service_excerpt,
                    kind=OfferKind.SERVICE,
                    jobs=("Купить programmatic-размещение",),
                ),
            ),
        )

        self.assertEqual(len(catalog.accepted_offers), 2)
        by_name = {item.canonical_name: item for item in catalog.accepted_offers}
        self.assertFalse(by_name["AdTech Compass"].generic_category_term)
        self.assertTrue(by_name["programmatic"].generic_category_term)
        self.assertTrue(
            by_name["programmatic"].evidence_refs[0].client_binding_proven
        )
        self.assertEqual(
            catalog.legacy_product_strings(),
            tuple(item.canonical_name for item in catalog.accepted_offers),
        )
        catalog.validate()

    def test_bare_topic_words_are_rejected_instead_of_inflating_visibility(self) -> None:
        topic_excerpt = "Запрос пользователя: DOOH и performance marketing. Сравните агентства."
        candidates = (
            _candidate(
                self.source,
                name="DOOH",
                excerpt=topic_excerpt,
                kind=OfferKind.SERVICE,
            ),
            _candidate(
                self.source,
                name="performance marketing",
                excerpt=topic_excerpt,
                kind=OfferKind.PRODUCT,
            ),
        )

        catalog = build_offer_catalog(
            client_domain="realweb.ru",
            client_aliases=("Realweb",),
            source_units=(self.source,),
            candidates=candidates,
        )

        self.assertEqual(catalog.accepted_offers, ())
        reasons = {item.reason for item in catalog.dispositions}
        self.assertIn("generic_category_term_without_client_offer_binding", reasons)
        self.assertIn("generic_category_term_is_not_a_proprietary_product", reasons)

    def test_explicitly_bound_generic_product_label_is_normalized_to_service(self) -> None:
        excerpt = "Realweb предлагает услугу programmatic для крупных брендов."
        catalog = build_offer_catalog(
            client_domain="realweb.ru",
            client_aliases=("Realweb",),
            source_units=(self.source,),
            candidates=(
                _candidate(
                    self.source,
                    name="programmatic",
                    excerpt=excerpt,
                    kind=OfferKind.PRODUCT,
                ),
            ),
        )

        self.assertEqual(catalog.accepted_offers[0].kind, OfferKind.SERVICE)
        self.assertEqual(
            catalog.dispositions[0].reason,
            "source_bound_commercial_offer_kind_normalized_to_service",
        )

    def test_generic_shared_alias_does_not_merge_distinct_branded_products(self) -> None:
        text = (
            "Realweb предлагает Brand Alpha для programmatic. "
            "Realweb предлагает Brand Beta для programmatic."
        )
        source = SourceUnit.from_text(
            source_unit_id="page:products",
            source_url="https://realweb.ru/products",
            text=text,
        )
        catalog = build_offer_catalog(
            client_domain="realweb.ru",
            client_aliases=("Realweb",),
            source_units=(source,),
            candidates=(
                _candidate(
                    source,
                    name="Brand Alpha",
                    aliases=("programmatic",),
                    excerpt="Realweb предлагает Brand Alpha для programmatic.",
                ),
                _candidate(
                    source,
                    name="Brand Beta",
                    aliases=("programmatic",),
                    excerpt="Realweb предлагает Brand Beta для programmatic.",
                ),
            ),
        )

        self.assertEqual(len(catalog.accepted_offers), 2)

    def test_exact_excerpt_source_url_and_digest_are_enforced(self) -> None:
        valid = _candidate(
            self.source,
            name="AdTech Compass",
            excerpt="Realweb развивает платформу AdTech Compass для медиапланирования.",
        )
        invalid = replace(
            valid,
            evidence_excerpt="Realweb развивает AdTech Compass.",
        )

        catalog = build_offer_catalog(
            client_domain="realweb.ru",
            client_aliases=("Realweb",),
            source_units=(self.source,),
            candidates=(invalid,),
        )

        self.assertEqual(catalog.accepted_offers, ())
        self.assertEqual(
            catalog.dispositions[0].reason,
            "excerpt_not_found_verbatim_in_source",
        )

        with self.assertRaisesRegex(OfferEvidenceError, "digest mismatch"):
            SourceUnit(
                source_unit_id="tampered",
                source_url="https://realweb.ru/",
                text="changed",
                source_sha256=self.source.source_sha256,
            )

    def test_duplicate_candidates_merge_and_keep_all_exact_evidence(self) -> None:
        first_excerpt = "Realweb развивает платформу AdTech Compass для медиапланирования."
        second_text = "Compass — продукт Realweb для агентских команд."
        second = SourceUnit.from_text(
            source_unit_id="page:compass",
            source_url="https://realweb.ru/compass",
            text=second_text,
        )
        catalog = build_offer_catalog(
            client_domain="realweb.ru",
            client_aliases=("Realweb",),
            source_units=(self.source, second),
            candidates=(
                _candidate(
                    self.source,
                    name="AdTech Compass",
                    aliases=("Compass",),
                    excerpt=first_excerpt,
                    confidence=0.95,
                ),
                _candidate(
                    second,
                    name="Compass",
                    aliases=("AdTech Compass",),
                    excerpt=second_text,
                    confidence=0.8,
                ),
            ),
        )

        self.assertEqual(len(catalog.accepted_offers), 1)
        self.assertEqual(len(catalog.accepted_offers[0].evidence_refs), 2)
        self.assertEqual(
            {item.decision.value for item in catalog.dispositions},
            {"accepted", "duplicate"},
        )

    def test_catalog_is_deterministic_under_candidate_and_source_reordering(self) -> None:
        excerpts = (
            "Realweb развивает платформу AdTech Compass для медиапланирования.",
            "Realweb предлагает услугу programmatic для крупных брендов.",
        )
        candidates = [
            _candidate(self.source, name="AdTech Compass", excerpt=excerpts[0]),
            _candidate(
                self.source,
                name="programmatic",
                excerpt=excerpts[1],
                kind=OfferKind.SERVICE,
            ),
        ]
        first = build_offer_catalog(
            client_domain="https://www.realweb.ru/",
            client_aliases=("Realweb",),
            source_units=(self.source,),
            candidates=candidates,
        )
        second = build_offer_catalog(
            client_domain="realweb.ru",
            client_aliases=("Realweb",),
            source_units=reversed((self.source,)),
            candidates=reversed(candidates),
        )

        self.assertEqual(first.as_dict(), second.as_dict())
        self.assertEqual(OfferCatalog.from_mapping(first.as_dict()).as_dict(), first.as_dict())


class OfferCandidateSourceNormalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.home = SourceUnit.from_text(
            source_unit_id="page:home",
            source_url="https://example.test/",
            text="Example представляет платформу Compass для медиапланирования.",
        )
        self.services = SourceUnit.from_text(
            source_unit_id="page:services",
            source_url="https://example.test/services",
            text="Example оказывает услугу аудита рекламных кампаний.",
        )

    def test_mapping_coordinates_are_repaired_only_for_unique_literal_excerpt(
        self,
    ) -> None:
        excerpt = "Example оказывает услугу аудита рекламных кампаний."
        raw = _candidate_mapping(
            self.home,
            name="Аудит рекламных кампаний",
            excerpt=excerpt,
        )
        raw.update(
            {
                "source_unit_id": "page:wrong",
                "source_url": "https://wrong.example/source",
                "source_sha256": "0" * 64,
            }
        )

        normalized, audit = normalize_offer_candidates_against_sources(
            source_units=(self.home, self.services),
            candidates=(raw,),
        )

        self.assertEqual(len(normalized), 1)
        candidate = normalized[0]
        self.assertEqual(candidate.source_unit_id, self.services.source_unit_id)
        self.assertEqual(candidate.source_url, self.services.source_url)
        self.assertEqual(candidate.source_sha256, self.services.source_sha256)
        self.assertEqual(candidate.evidence_excerpt, excerpt)
        self.assertEqual(candidate.canonical_name, "Аудит рекламных кампаний")
        self.assertEqual(audit["repaired_candidate_count"], 1)
        self.assertEqual(audit["malformed_candidate_count"], 0)
        self.assertEqual(audit["rows"][0]["status"], "repaired")
        self.assertEqual(
            audit["rows"][0]["repairs"],
            ["source_sha256", "source_unit_id", "source_url"],
        )

    def test_malformed_row_is_audited_without_discarding_other_candidates(
        self,
    ) -> None:
        first = _candidate_mapping(
            self.home,
            name="Compass",
            excerpt=self.home.text,
        )
        malformed = _candidate_mapping(
            self.home,
            name="Broken",
            excerpt=self.home.text,
        )
        malformed["aliases"] = "not-an-array"
        second = _candidate_mapping(
            self.services,
            name="Аудит рекламных кампаний",
            excerpt=self.services.text,
        )

        normalized, audit = normalize_offer_candidates_against_sources(
            source_units=(self.home, self.services),
            candidates=(first, malformed, second),
        )

        self.assertEqual(
            [candidate.canonical_name for candidate in normalized],
            ["Compass", "Аудит рекламных кампаний"],
        )
        self.assertEqual(audit["input_candidate_count"], 3)
        self.assertEqual(audit["valid_candidate_count"], 2)
        self.assertEqual(audit["malformed_candidate_count"], 1)
        self.assertEqual(
            [row["status"] for row in audit["rows"]],
            ["valid", "malformed", "valid"],
        )
        self.assertIn("aliases", audit["rows"][1]["error"])

    def test_ambiguous_literal_excerpt_is_not_rebound_or_admitted(self) -> None:
        excerpt = "Example предлагает общий пакет аналитики."
        first = SourceUnit.from_text(
            source_unit_id="page:first",
            source_url="https://example.test/first",
            text=excerpt,
        )
        second = SourceUnit.from_text(
            source_unit_id="page:second",
            source_url="https://example.test/second",
            text=excerpt,
        )
        raw = _candidate_mapping(first, name="Пакет аналитики", excerpt=excerpt)
        raw.update(
            {
                "source_unit_id": "page:missing",
                "source_url": "https://example.test/missing",
                "source_sha256": "f" * 64,
            }
        )

        normalized, audit = normalize_offer_candidates_against_sources(
            source_units=(first, second),
            candidates=(raw,),
        )

        self.assertEqual(normalized, ())
        self.assertEqual(audit["input_candidate_count"], 1)
        self.assertEqual(audit["valid_candidate_count"], 0)

    def test_offer_candidate_object_uses_the_same_source_binding_control(
        self,
    ) -> None:
        original = _candidate(
            self.services,
            name="Аудит рекламных кампаний",
            excerpt=self.services.text,
        )
        stale = replace(
            original,
            source_unit_id="page:stale",
            source_url="https://wrong.example/stale",
            source_sha256="f" * 64,
        )

        normalized, audit = normalize_offer_candidates_against_sources(
            source_units=(self.home, self.services),
            candidates=(stale,),
        )

        self.assertEqual(len(normalized), 1)
        candidate = normalized[0]
        self.assertEqual(candidate.source_unit_id, self.services.source_unit_id)
        self.assertEqual(candidate.source_url, self.services.source_url)
        self.assertEqual(candidate.source_sha256, self.services.source_sha256)
        self.assertEqual(audit["rows"][0]["status"], "repaired")
        self.assertEqual(
            audit["rows"][0]["repairs"],
            ["source_sha256", "source_unit_id", "source_url"],
        )


class OfferCatalogScopeTests(unittest.TestCase):
    def test_hard_maximum_10_dispositions_overflow_without_losing_audit(self) -> None:
        lines = [f"Realweb предлагает Product {index} бизнесу." for index in range(12)]
        source = SourceUnit.from_text(
            source_unit_id="page:portfolio",
            source_url="https://realweb.ru/portfolio",
            text="\n".join(lines),
        )
        candidates = [
            _candidate(
                source,
                name=f"Product {index}",
                excerpt=line,
                confidence=1.0 - index / 100,
                jobs=(f"Решить задачу {index}",),
            )
            for index, line in enumerate(lines)
        ]
        random.Random(17).shuffle(candidates)

        catalog = build_offer_catalog(
            client_domain="realweb.ru",
            client_aliases=("Realweb",),
            source_units=(source,),
            candidates=candidates,
        )

        self.assertEqual(len(catalog.accepted_offers), MAX_ACCEPTED_OFFERS)
        self.assertEqual(
            sum(item.decision.value == "overflow" for item in catalog.dispositions),
            2,
        )
        self.assertEqual(len(catalog.dispositions), 12)
        self.assertEqual(
            {item.canonical_name for item in catalog.accepted_offers},
            {f"Product {index}" for index in range(10)},
        )

    def test_caller_cannot_raise_offer_scope_above_ten(self) -> None:
        with self.assertRaisesRegex(OfferCatalogError, "1 to 10"):
            build_offer_catalog(
                client_domain="realweb.ru",
                client_aliases=("Realweb",),
                source_units=(),
                candidates=(),
                max_offers=11,
            )


class DomainResearchPayloadTests(unittest.TestCase):
    def test_large_evidence_is_sharded_and_reconstructed_without_tail_loss(self) -> None:
        tail = "TAIL::НЕ_ПОТЕРЯТЬ::🧿"
        excerpt = "Realweb предлагает Long Product. " + ("Детальное доказательство. " * 3_000) + tail
        source = SourceUnit.from_text(
            source_unit_id="page:long",
            source_url="https://realweb.ru/long",
            text=excerpt,
        )
        catalog = build_offer_catalog(
            client_domain="realweb.ru",
            client_aliases=("Realweb",),
            source_units=(source,),
            candidates=(
                _candidate(
                    source,
                    name="Long Product",
                    excerpt=excerpt,
                    jobs=("Изучить длинное доказательство",),
                ),
            ),
        )

        payload = build_domain_research_payload(catalog, target_utf8_bytes=257)
        reconstructed = reconstruct_domain_research_payload(payload)

        self.assertGreater(len(payload.shards), 100)
        self.assertTrue(
            reconstructed["offers"][0]["evidence_refs"][0]["evidence_excerpt"].endswith(tail)
        )
        self.assertEqual(
            reconstructed["offers"][0]["offer_id"],
            catalog.accepted_offers[0].offer_id,
        )
        self.assertEqual(
            sum(shard.utf8_length for shard in payload.shards),
            payload.manifest["document_utf8_length"],
        )

    def test_shard_tampering_is_detected(self) -> None:
        source = SourceUnit.from_text(
            source_unit_id="page:one",
            source_url="https://example.test/",
            text="Example предлагает Product One.",
        )
        catalog = build_offer_catalog(
            client_domain="example.test",
            client_aliases=("Example",),
            source_units=(source,),
            candidates=(
                _candidate(
                    source,
                    name="Product One",
                    excerpt=source.text,
                ),
            ),
        )
        payload = build_domain_research_payload(catalog, target_utf8_bytes=31)
        tampered_shard = replace(payload.shards[0], text=payload.shards[0].text + "x")
        tampered = replace(payload, shards=(tampered_shard, *payload.shards[1:]))

        with self.assertRaisesRegex(OfferCatalogError, "content identity"):
            reconstruct_domain_research_payload(tampered)


class PromptFoundationTests(unittest.TestCase):
    def setUp(self) -> None:
        text = (
            "Example предлагает Product One для выбора подрядчика.\n"
            "Example развивает Product Two для планирования кампании."
        )
        source = SourceUnit.from_text(
            source_unit_id="page:products",
            source_url="https://example.test/products",
            text=text,
        )
        self.catalog = build_offer_catalog(
            client_domain="example.test",
            client_aliases=("Example",),
            source_units=(source,),
            candidates=(
                _candidate(
                    source,
                    name="Product One",
                    excerpt="Example предлагает Product One для выбора подрядчика.",
                    jobs=("Выбрать подрядчика",),
                ),
                _candidate(
                    source,
                    name="Product Two",
                    excerpt="Example развивает Product Two для планирования кампании.",
                    jobs=("Спланировать кампанию",),
                ),
            ),
        )
        self.upstream = build_upstream_artifact_digests(
            profile={"brand": "Example"},
            catalog=self.catalog,
            market_research={"market": "advertising"},
            selected_pages_manifest={"urls": [source.source_url]},
        )
        self.clusters = build_offer_clusters(self.catalog)

    def test_six_intent_prompts_cover_every_offer_and_job(self) -> None:
        foundation = build_prompt_foundation(
            upstream=self.upstream,
            catalog=self.catalog,
            clusters=self.clusters,
            prompts=_six_prompts(self.clusters),
        )

        self.assertEqual(tuple(item.intent for item in foundation.prompts), INTENT_CODES)
        self.assertEqual(
            {offer_id for row in foundation.coverage for offer_id in row["offer_ids"]},
            {offer.offer_id for offer in self.catalog.accepted_offers},
        )
        self.assertTrue(all(row["supporting_prompt_keys"] for row in foundation.coverage))
        foundation.validate()

    def test_uncovered_cluster_fails_closed(self) -> None:
        with self.assertRaisesRegex(PromptCoverageError, "no supporting prompt"):
            build_prompt_foundation(
                upstream=self.upstream,
                catalog=self.catalog,
                clusters=self.clusters,
                prompts=_six_prompts(self.clusters, cover=False),
            )

    def test_explicit_specific_exclusion_accounts_for_cluster(self) -> None:
        exclusions = tuple(
            ClusterExclusion(
                cluster_id=cluster.cluster_id,
                reason="Исключено из шести сценариев: услуга доступна только действующим клиентам.",
            )
            for cluster in self.clusters
        )
        foundation = build_prompt_foundation(
            upstream=self.upstream,
            catalog=self.catalog,
            clusters=self.clusters,
            prompts=_six_prompts(self.clusters, cover=False),
            exclusions=exclusions,
        )

        self.assertTrue(all(row["exclusion_reason"] for row in foundation.coverage))
        with self.assertRaises(PromptCoverageError):
            ClusterExclusion(cluster_id=self.clusters[0].cluster_id, reason="N/A")

    def test_custom_cluster_assignments_must_partition_catalog(self) -> None:
        first_offer = self.catalog.accepted_offers[0]
        with self.assertRaisesRegex(PromptCoverageError, "omit offers"):
            build_offer_clusters(
                self.catalog,
                assignments={"first": [first_offer.offer_id]},
            )


class ResumeCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        source = SourceUnit.from_text(
            source_unit_id="page:product",
            source_url="https://example.test/product",
            text="Example предлагает Product One.",
        )
        self.catalog = build_offer_catalog(
            client_domain="example.test",
            client_aliases=("Example",),
            source_units=(source,),
            candidates=(
                _candidate(source, name="Product One", excerpt=source.text),
            ),
        )
        self.upstream = build_upstream_artifact_digests(
            profile={"brand": "Example", "version": 1},
            catalog=self.catalog,
            market_research={"market": "one"},
            selected_pages_manifest={"urls": [source.source_url]},
        )
        clusters = build_offer_clusters(self.catalog)
        self.foundation = build_prompt_foundation(
            upstream=self.upstream,
            catalog=self.catalog,
            clusters=clusters,
            prompts=_six_prompts(clusters),
        )
        self.answers = {
            key: {"ChatGPT": f"Ответ для {key}"}
            for key in self.foundation.prompt_keys
        }
        self.receipt = build_answer_set_receipt(self.foundation, self.answers)

    def test_matching_artifacts_can_resume(self) -> None:
        report = validate_resume_compatibility(
            current_upstream=self.upstream,
            persisted_foundation=self.foundation,
            answer_receipt=self.receipt,
            persisted_answers_by_prompt=self.answers,
        )

        self.assertTrue(report.compatible)
        self.assertEqual(report.mismatches, ())

        serialized_foundation = self.foundation.as_dict()
        parsed_foundation = PromptFoundation.from_mapping(serialized_foundation)
        mapped_report = validate_resume_compatibility(
            current_upstream=self.upstream.as_dict(),
            persisted_foundation=parsed_foundation.as_dict(),
            answer_receipt=self.receipt.as_dict(),
            persisted_answers_by_prompt=self.answers,
        )
        self.assertTrue(mapped_report.compatible)

    def test_new_profile_cannot_reuse_old_prompts_or_answers(self) -> None:
        changed = replace(
            self.upstream,
            profile_digest=artifact_digest({"brand": "Example", "version": 2}),
        )

        report = audit_resume_compatibility(
            current_upstream=changed,
            persisted_foundation=self.foundation,
            answer_receipt=self.receipt,
            persisted_answers_by_prompt=self.answers,
        )
        self.assertFalse(report.compatible)
        self.assertTrue(
            any(item.startswith("profile_digest:") for item in report.mismatches)
        )
        with self.assertRaisesRegex(ResumeCompatibilityError, "profile_digest"):
            validate_resume_compatibility(
                current_upstream=changed,
                persisted_foundation=self.foundation,
                answer_receipt=self.receipt,
                persisted_answers_by_prompt=self.answers,
            )

    def test_new_catalog_digest_cannot_reuse_old_prompt_foundation(self) -> None:
        changed = replace(
            self.upstream,
            catalog_digest=hashlib.sha256(b"new-catalog").hexdigest(),
        )

        with self.assertRaisesRegex(ResumeCompatibilityError, "catalog_digest"):
            validate_resume_compatibility(
                current_upstream=changed,
                persisted_foundation=self.foundation,
                answer_receipt=self.receipt,
                persisted_answers_by_prompt=self.answers,
            )

    def test_answer_content_tampering_breaks_receipt(self) -> None:
        changed_answers = dict(self.answers)
        changed_answers[self.foundation.prompt_keys[-1]] = {"ChatGPT": "Подменённый ответ"}

        with self.assertRaisesRegex(
            ResumeCompatibilityError,
            "answer_set_content_digest_mismatch",
        ):
            validate_resume_compatibility(
                current_upstream=self.upstream,
                persisted_foundation=self.foundation,
                answer_receipt=self.receipt,
                persisted_answers_by_prompt=changed_answers,
            )

    def test_answers_without_receipt_are_not_reusable(self) -> None:
        with self.assertRaisesRegex(
            ResumeCompatibilityError,
            "persisted_answers_have_no_answer_receipt",
        ):
            validate_resume_compatibility(
                current_upstream=self.upstream,
                persisted_foundation=self.foundation,
                persisted_answers_by_prompt=self.answers,
            )

    def test_answer_receipt_requires_exact_six_prompt_keys(self) -> None:
        incomplete = dict(self.answers)
        incomplete.pop(self.foundation.prompt_keys[0])

        with self.assertRaisesRegex(ResumeCompatibilityError, "missing"):
            build_answer_set_receipt(self.foundation, incomplete)

        forged = AnswerSetReceipt(
            prompt_foundation_digest=self.receipt.prompt_foundation_digest,
            prompt_set_digest=self.receipt.prompt_set_digest,
            prompt_keys=self.receipt.prompt_keys,
            answer_set_digest="bad",
        )
        report = audit_resume_compatibility(
            current_upstream=self.upstream,
            persisted_foundation=self.foundation,
            answer_receipt=forged,
        )
        self.assertFalse(report.compatible)


class DigestContractTests(unittest.TestCase):
    def test_artifact_digest_is_key_order_independent_and_content_sensitive(self) -> None:
        self.assertEqual(
            artifact_digest({"b": 2, "a": 1}),
            artifact_digest({"a": 1, "b": 2}),
        )
        self.assertNotEqual(
            artifact_digest({"a": 1}),
            artifact_digest({"a": 2}),
        )

    def test_upstream_digest_dataclass_rejects_non_sha_values(self) -> None:
        with self.assertRaisesRegex(OfferCatalogError, "profile_digest"):
            UpstreamArtifactDigests(
                profile_digest="bad",
                catalog_digest="0" * 64,
                market_research_digest="0" * 64,
                selected_pages_digest="0" * 64,
            )


if __name__ == "__main__":
    unittest.main()
