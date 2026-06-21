import json
import re
import unittest
from pathlib import Path


def _css_rule(html: str, selector: str) -> str:
    match = re.search(rf"{re.escape(selector)}\s*\{{([^}}]+)\}}", html)
    if match is None:
        raise AssertionError(f"missing CSS rule for {selector}")
    return " ".join(match.group(1).split())


class FrontendStaticTests(unittest.TestCase):
    def test_report_markdown_uses_full_panel_width(self) -> None:
        html = Path("static/index.html").read_text(encoding="utf-8")

        prose_rule = _css_rule(html, ".prose-rwplus")
        table_rule = _css_rule(html, ".prose-rwplus table")

        self.assertIn('class="prose prose-rwplus max-w-none"', html)
        self.assertIn("max-width: none", prose_rule)
        self.assertIn("width: 100%", prose_rule)
        self.assertIn("width: 100%", table_rule)

    def test_homepage_explains_ai_visibility_methodology(self) -> None:
        html = Path("static/index.html").read_text(encoding="utf-8")
        html_lower = html.lower()

        self.assertIn("<details", html)
        self.assertIn("<summary", html)
        self.assertIn("var(--brand-info)", html)
        self.assertNotIn("box-shadow: inset 4px 0 0 var(--brand-primary)", html)
        self.assertIn("Оценка AI visibility паблишеров и web-ресурсов с учетом технической доступности для LLM", html)
        self.assertIn("реальные отпечатки AI-ботов", html)
        self.assertIn("<strong>реальные отпечатки AI-ботов</strong>", html)
        self.assertIn("GPTBot", html)
        self.assertIn("OAI-SearchBot", html)
        self.assertIn("ClaudeBot", html)
        self.assertIn("PerplexityBot", html)
        self.assertIn("Accept-Language", html)
        self.assertIn("источников знаний LLM-систем", html)
        self.assertIn("AI-видимость продуктов и брендов", html)
        self.assertIn("<strong>AI-видимость продуктов и брендов</strong>", html)
        self.assertIn("web search tool", html)
        self.assertIn("RAG", html)
        self.assertIn("retrieval", html)
        self.assertIn("rerank", html)
        self.assertIn("сетевые ограничения в россии", html_lower)
        self.assertIn("Chrome-control", html)
        self.assertIn("почему результату можно доверять", html_lower)

    def test_manual_research_set_description_describes_landscape_baseline(self) -> None:
        data = json.loads(Path("data/sets/set1_manual_research.json").read_text(encoding="utf-8"))

        self.assertIn("базовый набор доменов", data["description"])
        self.assertIn("ландшафта доступности", data["description"])
        self.assertIn("AI-видимости источников", data["description"])
        self.assertNotIn("web_search/web_fetch", data["description"])

    def test_runwai_mobile_header_can_wrap_navigation(self) -> None:
        css = Path("static/rwy.css").read_text(encoding="utf-8")

        self.assertIn("@media (max-width: 720px)", css)
        self.assertIn("header > div.flex.items-center.gap-4", css)
        self.assertIn("flex-direction: column !important", css)
        self.assertIn("header nav.flex.gap-1.text-sm", css)
        self.assertIn("grid-template-columns: repeat(2, minmax(0, 1fr))", css)
        self.assertIn("white-space: normal", css)


if __name__ == "__main__":
    unittest.main()
