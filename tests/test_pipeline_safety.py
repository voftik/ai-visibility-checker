import unittest
import uuid
from pathlib import Path

from sqlalchemy import delete, func, select, text

from app.db import SessionLocal
from app.models import DomainProbe, ProbeType, RobotsRule, Run, RunStatus
from app.services import crawler
from app.services.analyzer import _build_dataset_text
from app.services.robots_parser import parse_robots


def _rules_by_bot(text: str) -> dict[str, tuple[str, str]]:
    return {bot: (rule, raw) for bot, rule, raw in parse_robots(text)}


class RobotsParserTests(unittest.TestCase):
    def test_wildcard_group_applies_to_known_bots(self) -> None:
        rules = _rules_by_bot(
            """
            User-agent: *
            Disallow: /
            """
        )

        self.assertEqual(rules["GPTBot"][0], "disallow_all")
        self.assertEqual(rules["ClaudeBot"][0], "disallow_all")
        self.assertEqual(rules["*"][0], "disallow_all")
        self.assertIn("User-agent: *", rules["GPTBot"][1])

    def test_specific_bot_group_overrides_wildcard(self) -> None:
        rules = _rules_by_bot(
            """
            User-agent: *
            Disallow: /

            User-agent: GPTBot
            Allow: /
            """
        )

        self.assertEqual(rules["GPTBot"][0], "allow_all")
        self.assertEqual(rules["ClaudeBot"][0], "disallow_all")


class CrawlerSafetyTests(unittest.TestCase):
    def test_default_headers_do_not_advertise_unsupported_brotli(self) -> None:
        encodings = {
            token.strip().lower()
            for token in crawler.DEFAULT_HEADERS["Accept-Encoding"].split(",")
        }

        self.assertNotIn("br", encodings)

    def test_build_jobs_normalizes_and_deduplicates_domains(self) -> None:
        jobs = crawler._build_jobs(
            [" https://WWW.Example.com/path?q=1 ", "example.com", ""],
            ["GPTBot", "unknown-ua"],
        )

        self.assertEqual(len(jobs), 2)
        self.assertEqual([j.domain for j in jobs], ["example.com", "example.com"])
        self.assertEqual(jobs[0].target_url, "https://example.com/robots.txt")
        self.assertEqual(jobs[1].target_url, "https://example.com/")


class DatabaseSafetyTests(unittest.IsolatedAsyncioTestCase):
    async def test_sqlite_foreign_keys_are_enabled(self) -> None:
        async with SessionLocal() as session:
            enabled = (await session.execute(text("PRAGMA foreign_keys"))).scalar_one()

        self.assertEqual(enabled, 1)

    async def test_run_delete_cascades_to_probe_and_robot_rows(self) -> None:
        run_id = f"test-{uuid.uuid4()}"
        async with SessionLocal() as session:
            try:
                session.add(
                    Run(
                        id=run_id,
                        status=RunStatus.completed,
                        config_json={},
                        progress_current=1,
                        progress_total=1,
                    )
                )
                session.add(
                    DomainProbe(
                        run_id=run_id,
                        domain="example.com",
                        user_agent_label="GPTBot",
                        user_agent_string="GPTBot",
                        target_url="https://example.com/",
                        probe_type=ProbeType.main_page,
                        challenge_detected=False,
                        body_looks_empty=False,
                    )
                )
                session.add(
                    RobotsRule(
                        run_id=run_id,
                        domain="example.com",
                        bot_name="GPTBot",
                        rule="allow_all",
                    )
                )
                await session.commit()

                await session.execute(delete(Run).where(Run.id == run_id))
                await session.commit()

                probes_left = (
                    await session.execute(
                        select(func.count()).select_from(DomainProbe).where(DomainProbe.run_id == run_id)
                    )
                ).scalar_one()
                rules_left = (
                    await session.execute(
                        select(func.count()).select_from(RobotsRule).where(RobotsRule.run_id == run_id)
                    )
                ).scalar_one()
            finally:
                await session.execute(delete(DomainProbe).where(DomainProbe.run_id == run_id))
                await session.execute(delete(RobotsRule).where(RobotsRule.run_id == run_id))
                await session.execute(delete(Run).where(Run.id == run_id))
                await session.commit()

        self.assertEqual(probes_left, 0)
        self.assertEqual(rules_left, 0)


class AnalyzerDatasetSafetyTests(unittest.IsolatedAsyncioTestCase):
    async def test_large_all_options_dataset_stays_under_llm_budget(self) -> None:
        run_id = f"test-{uuid.uuid4()}"
        domains = [f"large-{i}.example" for i in range(100)]
        user_agents = [label for label in crawler.USER_AGENT_STRINGS if label != "robots-fetcher"]
        bots = [
            "GPTBot",
            "OAI-SearchBot",
            "ChatGPT-User",
            "ClaudeBot",
            "anthropic-ai",
            "Claude-Web",
            "PerplexityBot",
            "Perplexity-User",
            "DeepSeekBot",
            "DeepSeek-User",
            "Google-Extended",
            "Googlebot",
            "GoogleOther",
            "Google-Agent",
            "Google-NotebookLM",
            "Google-CloudVertexBot",
            "Gemini-Deep-Research",
            "Applebot-Extended",
            "CCBot",
            "Bytespider",
            "*",
        ]

        async with SessionLocal() as session:
            try:
                session.add(
                    Run(
                        id=run_id,
                        status=RunStatus.completed,
                        config_json={
                            "domains": domains,
                            "user_agents": user_agents,
                            "concurrency": 8,
                            "timeout_seconds": 15,
                        },
                        progress_current=len(domains) * (len(user_agents) + 1),
                        progress_total=len(domains) * (len(user_agents) + 1),
                    )
                )
                for domain in domains:
                    session.add(
                        DomainProbe(
                            run_id=run_id,
                            domain=domain,
                            user_agent_label="robots-fetcher",
                            user_agent_string="ai-visibility-checker/1.0 research bot",
                            target_url=f"https://{domain}/robots.txt",
                            probe_type=ProbeType.robots_txt,
                            http_status=200,
                            response_size_bytes=1500,
                            total_time_ms=120,
                            tls_ok=True,
                            final_url=f"https://{domain}/robots.txt",
                            challenge_detected=False,
                            body_looks_empty=False,
                        )
                    )
                    for bot in bots:
                        session.add(
                            RobotsRule(
                                run_id=run_id,
                                domain=domain,
                                bot_name=bot,
                                rule="partial",
                                raw_directives=(
                                    f"User-agent: {bot}\n"
                                    "Disallow: /private\n"
                                    "Allow: /public\n"
                                    "Crawl-delay: 1"
                                ),
                            )
                        )
                    for label in user_agents:
                        session.add(
                            DomainProbe(
                                run_id=run_id,
                                domain=domain,
                                user_agent_label=label,
                                user_agent_string=crawler.USER_AGENT_STRINGS[label],
                                target_url=f"https://{domain}/",
                                probe_type=ProbeType.main_page,
                                http_status=403,
                                response_size_bytes=12000,
                                total_time_ms=500,
                                tls_ok=True,
                                final_url=f"https://{domain}/",
                                detected_protections=["cloudflare", "ua-conditional-block"],
                                challenge_detected=False,
                                body_looks_empty=False,
                                content_extractable_text_length=100,
                                content_signals={
                                    "looks_like_error_page": True,
                                    "primary_language_guess": "ru",
                                },
                            )
                        )
                await session.commit()

                _, meta = await _build_dataset_text(run_id)
            finally:
                await session.execute(delete(DomainProbe).where(DomainProbe.run_id == run_id))
                await session.execute(delete(RobotsRule).where(RobotsRule.run_id == run_id))
                await session.execute(delete(Run).where(Run.id == run_id))
                await session.commit()

        self.assertLessEqual(meta["char_count"], 250_000)


class FrontendSafetyTests(unittest.TestCase):
    def test_start_button_waits_for_enabled_sets_to_finish_loading(self) -> None:
        html = Path("static/index.html").read_text(encoding="utf-8")

        self.assertIn("get setsLoading()", html)
        self.assertIn("setsLoading || submitting", html)
        self.assertIn("if (this.setsLoading)", html)


if __name__ == "__main__":
    unittest.main()
