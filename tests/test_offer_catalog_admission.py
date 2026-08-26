from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from app.services.analyzer import _bind_offer_catalog
from app.services.offer_catalog import (
    INTENT_CODES,
    IntentPrompt,
    OfferCatalogAdmission,
    OfferCatalogAdmissionError,
    OfferCatalogAdmissionStatus,
    PromptFoundation,
    SourceUnit,
    ZeroOfferSiteKind,
    assess_offer_catalog_admission,
    build_offer_catalog,
    build_offer_clusters,
    build_prompt_foundation,
    build_upstream_artifact_digests,
)


def _empty_catalog(
    text: str = "Сохранённый текст главной страницы.",
):
    source = SourceUnit.from_text(
        source_unit_id="page:home",
        source_url="https://example.test/",
        text=text,
    )
    catalog = build_offer_catalog(
        client_domain="example.test",
        client_aliases=("Example",),
        source_units=(source,),
        candidates=(),
    )
    return catalog, source


def _six_generic_prompts() -> tuple[IntentPrompt, ...]:
    return tuple(
        IntentPrompt(
            prompt_key=f"prompt:{intent.casefold()}",
            intent=intent,
            text=f"Пользовательский сценарий {intent}",
            supporting_cluster_ids=(),
        )
        for intent in INTENT_CODES
    )


class ZeroOfferAdmissionTests(unittest.TestCase):
    def _upstream(self, profile, *, source_text: str):
        catalog, source = _empty_catalog(source_text)
        upstream = build_upstream_artifact_digests(
            profile=profile,
            catalog=catalog,
            market_research={"status": "ready"},
            selected_pages_manifest={"urls": [source.source_url]},
        )
        return catalog, upstream, source

    def test_commercial_zero_catalog_is_rejected_before_generic_intents(
        self,
    ) -> None:
        profile = {
            "site_type": "Сайт продукта",
            "business_model": "B2B-подписка",
            "products": ["Atlas"],
            "offer_candidates": [
                {
                    "canonical_name": "Atlas",
                    "commercially_relevant": True,
                }
            ],
            "entity_scope": [],
            "evidence": [
                "На странице продукт назван коммерческим решением."
            ],
            "confidence": "high",
        }
        catalog, upstream, source = self._upstream(
            profile,
            source_text="На странице продукт назван коммерческим решением.",
        )

        admission = assess_offer_catalog_admission(
            catalog=catalog,
            profile=profile,
            source_units=(source,),
        )

        self.assertEqual(
            admission.status,
            OfferCatalogAdmissionStatus.ZERO_OFFERS_BLOCKED,
        )
        self.assertEqual(
            admission.site_kind,
            ZeroOfferSiteKind.COMMERCIAL_OR_UNKNOWN,
        )
        self.assertIn("commercial_offer_signals_present", admission.reason_codes)
        with self.assertRaisesRegex(
            OfferCatalogAdmissionError,
            "generic INTENT generation is blocked",
        ):
            build_prompt_foundation(
                upstream=upstream,
                catalog=catalog,
                clusters=build_offer_clusters(catalog),
                prompts=_six_generic_prompts(),
                profile=profile,
                source_units=(source,),
            )

    def test_genuine_noncommercial_zero_catalog_is_admitted(self) -> None:
        profile = {
            "site_type": "Некоммерческий информационный ресурс",
            "business_model": "Некоммерческий проект",
            "products": [],
            "offer_candidates": [],
            "entity_scope": [],
            "evidence": [
                "Это некоммерческий проект с общественной "
                "просветительской миссией."
            ],
            "confidence": "high",
        }
        catalog, upstream, source = self._upstream(
            profile,
            source_text=(
                "Это некоммерческий проект с общественной "
                "просветительской миссией."
            ),
        )

        admission = assess_offer_catalog_admission(
            catalog=catalog,
            profile=profile,
            source_units=(source,),
            source_absence_claims_allowed=True,
            crawl_coverage_state="complete",
        )
        foundation = build_prompt_foundation(
            upstream=upstream,
            catalog=catalog,
            clusters=build_offer_clusters(catalog),
            prompts=_six_generic_prompts(),
            profile=profile,
            source_units=(source,),
            source_absence_claims_allowed=True,
            crawl_coverage_state="complete",
        )

        self.assertEqual(
            admission.status,
            OfferCatalogAdmissionStatus.ZERO_OFFERS_ADMITTED,
        )
        self.assertEqual(admission.site_kind, ZeroOfferSiteKind.NONCOMMERCIAL)
        self.assertTrue(admission.allowed)
        self.assertEqual(len(admission.source_evidence_receipts), 1)
        self.assertEqual(
            admission.source_evidence_receipts[0]["source_unit_id"],
            source.source_unit_id,
        )
        self.assertEqual(
            admission.source_evidence_receipts[0]["source_sha256"],
            source.source_sha256,
        )
        self.assertEqual(
            OfferCatalogAdmission.from_mapping(admission.as_dict()),
            admission,
        )
        self.assertEqual(foundation.clusters, ())
        self.assertEqual(len(foundation.prompts), len(INTENT_CODES))
        # The v1 persisted foundation shape stays readable. Admission is a
        # preflight contract, not a new required field in saved artifacts.
        self.assertEqual(
            PromptFoundation.from_mapping(foundation.as_dict()).foundation_digest,
            foundation.foundation_digest,
        )

    def test_plausible_but_nonliteral_evidence_does_not_admit_zero(self) -> None:
        profile = {
            "site_type": "Некоммерческий информационный ресурс",
            "business_model": "Некоммерческий проект",
            "products": [],
            "offer_candidates": [],
            "entity_scope": [],
            "evidence": [
                "Проект ведёт общественную просветительскую работу."
            ],
            "confidence": "high",
        }
        catalog, upstream, source = self._upstream(
            profile,
            source_text=(
                "На странице описаны новости, события и материалы проекта."
            ),
        )

        admission = assess_offer_catalog_admission(
            catalog=catalog,
            profile=profile,
            source_units=(source,),
        )

        self.assertFalse(admission.allowed)
        self.assertEqual(admission.source_evidence_receipts, ())
        self.assertIn(
            "missing_source_bound_model_evidence",
            admission.reason_codes,
        )
        self.assertIn(
            "missing_source_bound_site_kind_evidence",
            admission.reason_codes,
        )
        with self.assertRaises(OfferCatalogAdmissionError):
            build_prompt_foundation(
                upstream=upstream,
                catalog=catalog,
                clusters=build_offer_clusters(catalog),
                prompts=_six_generic_prompts(),
                profile=profile,
                source_units=(source,),
            )

    def test_unrelated_exact_quote_cannot_license_no_public_offer(self) -> None:
        profile = {
            "site_type": "Сайт без публичных предложений",
            "business_model": "",
            "products": [],
            "offer_candidates": [],
            "entity_scope": [],
            "evidence": ["Добро пожаловать на сайт компании."],
            "confidence": "high",
        }
        catalog, upstream, source = self._upstream(
            profile,
            source_text="Добро пожаловать на сайт компании.",
        )

        admission = assess_offer_catalog_admission(
            catalog=catalog,
            profile=profile,
            source_units=(source,),
        )

        self.assertFalse(admission.allowed)
        self.assertEqual(admission.source_evidence_receipts, ())
        self.assertIn(
            "missing_source_bound_site_kind_evidence",
            admission.reason_codes,
        )
        with self.assertRaises(OfferCatalogAdmissionError):
            build_prompt_foundation(
                upstream=upstream,
                catalog=catalog,
                clusters=build_offer_clusters(catalog),
                prompts=_six_generic_prompts(),
                profile=profile,
                source_units=(source,),
            )

    def test_full_source_corpus_can_contradict_zero_offer_claim(self) -> None:
        profile = {
            "site_type": "Некоммерческий проект",
            "business_model": "Некоммерческий проект",
            "products": [],
            "offer_candidates": [],
            "entity_scope": [],
            "evidence": ["Это некоммерческий проект."],
            "confidence": "high",
        }
        home = SourceUnit.from_text(
            source_unit_id="page:home",
            source_url="https://example.test/",
            text="Это некоммерческий проект.",
        )
        tickets = SourceUnit.from_text(
            source_unit_id="page:tickets",
            source_url="https://example.test/tickets",
            text="Купить билет и оплатить заказ можно на этой странице.",
        )
        catalog = build_offer_catalog(
            client_domain="example.test",
            client_aliases=("Example",),
            source_units=(home, tickets),
            candidates=(),
        )

        admission = assess_offer_catalog_admission(
            catalog=catalog,
            profile=profile,
            source_units=(home, tickets),
        )

        self.assertFalse(admission.allowed)
        self.assertTrue(admission.source_commercial_signal_receipts)
        self.assertIn(
            "source_commercial_offer_signals_present",
            admission.reason_codes,
        )

    def test_zero_offer_claim_requires_complete_absence_safe_crawl(self) -> None:
        profile = {
            "site_type": "Некоммерческий проект",
            "business_model": "Некоммерческий проект",
            "products": [],
            "offer_candidates": [],
            "entity_scope": [],
            "evidence": ["Это некоммерческий проект."],
            "confidence": "high",
        }
        catalog, _upstream, source = self._upstream(
            profile,
            source_text="Это некоммерческий проект.",
        )

        prefix_blocked = assess_offer_catalog_admission(
            catalog=catalog,
            profile=profile,
            source_units=(source,),
            source_absence_claims_allowed=False,
            crawl_coverage_state="complete",
        )
        bounded_blocked = assess_offer_catalog_admission(
            catalog=catalog,
            profile=profile,
            source_units=(source,),
            source_absence_claims_allowed=True,
            crawl_coverage_state="bounded",
        )

        self.assertFalse(prefix_blocked.allowed)
        self.assertIn(
            "source_absence_claims_not_allowed",
            prefix_blocked.reason_codes,
        )
        self.assertFalse(bounded_blocked.allowed)
        self.assertIn(
            "crawl_coverage_not_complete",
            bounded_blocked.reason_codes,
        )


class AnalyzerZeroOfferPreflightTests(unittest.IsolatedAsyncioTestCase):
    async def test_bind_uses_structured_homepage_identity_for_non_domain_brand(
        self,
    ) -> None:
        text = "Мы — Acme. Acme предлагает сервис Atlas для управления продажами."
        source = SourceUnit.from_text(
            source_unit_id="https://holding-company.com/",
            source_url="https://holding-company.com/",
            text=text,
        )
        profile = {
            "brand_name": "Acme",
            "brand_aliases": [],
            "site_type": "Сайт продукта",
            "business_model": "B2B-подписка",
            "products": ["Atlas"],
            "offer_candidates": [
                {
                    "canonical_name": "Atlas",
                    "aliases": [],
                    "kind": "service",
                    "source_url": source.source_url,
                    "evidence_excerpt": (
                        "Acme предлагает сервис Atlas для управления продажами."
                    ),
                    "source_unit_id": source.source_unit_id,
                    "source_sha256": source.source_sha256,
                    "confidence": 0.95,
                    "user_jobs": ["управления продажами"],
                    "commercially_relevant": True,
                }
            ],
            "entity_scope": [],
            "evidence": ["Acme предлагает сервис Atlas для управления продажами."],
            "uncertainties": [],
            "confidence": "high",
        }
        site_context = {
            "requested_site": {
                "domain": "holding-company.com",
                "url": source.source_url,
            },
            "pages": [
                {
                    "url": source.source_url,
                    "source_unit_id": source.source_unit_id,
                    "source_sha256": source.source_sha256,
                    "page_kind": "home",
                    "title": "Acme | Sales platform",
                    "main_text": text,
                    "absence_claims_allowed": True,
                }
            ],
            "crawl_admission": {"coverage_state": "complete"},
        }

        with patch(
            "app.services.analyzer._save_artifact",
            new_callable=AsyncMock,
        ) as save_artifact:
            enriched, catalog, _clusters = await _bind_offer_catalog(
                "run-id",
                profile,
                site_context,
            )

        self.assertEqual([offer.canonical_name for offer in catalog.accepted_offers], ["Atlas"])
        self.assertEqual(enriched["offer_catalog"]["client_aliases"], [
            "Acme",
            "holding-company",
            "holding-company.com",
        ])
        identity_call = next(
            call
            for call in save_artifact.await_args_list
            if call.kwargs["artifact_key"] == "client_identity_admission"
        )
        self.assertEqual(
            identity_call.kwargs["input_json"]["structured_identity_record_digest"],
            identity_call.kwargs["output_json"]["structured_identity_record_digest"],
        )

    async def test_bind_blocks_commercial_zero_before_intent_generation(
        self,
    ) -> None:
        profile = {
            "brand_name": "Example",
            "brand_aliases": [],
            "site_type": "Сайт продукта",
            "business_model": "B2B-подписка",
            "products": ["Atlas"],
            "offer_candidates": [],
            "entity_scope": [],
            "evidence": [
                "Главная страница описывает коммерческий продукт."
            ],
            "uncertainties": [],
            "confidence": "high",
        }
        site_context = {
            "requested_site": {
                "domain": "example.test",
                "url": "https://example.test/",
            },
            "pages": [
                {
                    "url": "https://example.test/",
                    "main_text": (
                        "Главная страница описывает коммерческий продукт. "
                        "Example описывает продукт Atlas."
                    ),
                }
            ],
        }

        with patch(
            "app.services.analyzer._save_artifact",
            new_callable=AsyncMock,
        ) as save_artifact:
            with self.assertRaises(OfferCatalogAdmissionError):
                await _bind_offer_catalog("run-id", profile, site_context)

        artifact_calls = {
            call.kwargs["artifact_key"]: call
            for call in save_artifact.await_args_list
        }
        self.assertEqual(
            set(artifact_calls),
            {
                "client_identity_admission",
                "offer_candidate_normalization_audit",
                "offer_catalog",
                "offer_catalog_admission",
                "offer_catalog_research_payload",
            },
        )
        self.assertFalse(
            artifact_calls["offer_catalog_admission"]
            .kwargs["output_json"]["allowed"]
        )
        self.assertEqual(
            artifact_calls["offer_catalog"].kwargs["output_json"]["accepted_offers"],
            [],
        )

    async def test_bind_admits_grounded_noncommercial_zero(self) -> None:
        profile = {
            "brand_name": "Example",
            "brand_aliases": [],
            "site_type": "Некоммерческий информационный ресурс",
            "business_model": "Некоммерческий проект",
            "products": [],
            "offer_candidates": [],
            "entity_scope": [],
            "evidence": [
                "Это некоммерческий проект с просветительской миссией."
            ],
            "uncertainties": [],
            "confidence": "high",
        }
        site_context = {
            "requested_site": {
                "domain": "example.test",
                "url": "https://example.test/",
            },
            "pages": [
                {
                    "url": "https://example.test/",
                    "main_text": (
                        "Это некоммерческий   проект с просветительской миссией."
                    ),
                    "absence_claims_allowed": True,
                }
            ],
            "crawl_admission": {"coverage_state": "complete"},
        }

        with patch(
            "app.services.analyzer._save_artifact",
            new_callable=AsyncMock,
        ) as save_artifact:
            enriched, catalog, clusters = await _bind_offer_catalog(
                "run-id",
                profile,
                site_context,
            )

        self.assertEqual(catalog.accepted_offers, ())
        self.assertEqual(clusters, ())
        self.assertEqual(enriched["products"], [])
        admission_call = next(
            call
            for call in save_artifact.await_args_list
            if call.kwargs["artifact_key"] == "offer_catalog_admission"
        )
        self.assertTrue(admission_call.kwargs["output_json"]["allowed"])


if __name__ == "__main__":
    unittest.main()
