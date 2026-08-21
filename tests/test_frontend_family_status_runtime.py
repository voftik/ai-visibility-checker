import subprocess
import unittest
from pathlib import Path


class FrontendFamilyStatusRuntimeTests(unittest.TestCase):
    """Run the shipped family-status renderer against binary and missing states."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.html = Path("static/index.html").read_text(encoding="utf-8")

    def _declaration(self, name: str) -> str:
        marker = f"      const {name} ="
        start = self.html.find(marker)
        self.assertNotEqual(start, -1, f"Missing frontend declaration: {name}")
        end = self.html.find("\n\n      const ", start + len(marker))
        self.assertNotEqual(end, -1, f"Could not find declaration end: {name}")
        return self.html[start:end].strip()

    def test_family_status_renderer_uses_lamps_tooltips_and_official_mark(self) -> None:
        declarations = "\n\n".join(
            self._declaration(name)
            for name in (
                "escapeHTML",
                "finiteNumber",
                "safeNumber",
                "auditTooltipHTML",
                "providerChartVisuals",
                "providerFallbackColors",
                "providerChartVisual",
                "providerIdentityHTML",
                "familyAccessStatus",
                "familyStatusPanelHTML",
            )
        )
        script = "\n".join(
            (
                '"use strict";',
                'const assert = require("node:assert/strict");',
                declarations,
                r'''
const available = familyAccessStatus({
  state: "available",
  passed_count: 1,
  total_count: 1,
  expected_count: 1
});
assert.equal(available.key, "available");
assert.equal(available.label, "Доступ открыт");
assert.match(available.tooltip, /возможность открыть страницы, но не.качество/);

const blocked = familyAccessStatus({
  state: "blocked",
  passed_count: 0,
  total_count: 1,
  expected_count: 1
});
assert.equal(blocked.key, "blocked");
assert.equal(blocked.label, "Доступ закрыт");
assert.match(blocked.tooltip, /ни.в.одной из.1 проверок/);

const partial = familyAccessStatus({
  state: "partial",
  passed_count: 1,
  total_count: 2,
  expected_count: 2
});
assert.equal(partial.key, "partial");
assert.equal(partial.label, "Не подтверждён");
assert.match(partial.tooltip, /доступ не.считается подтверждённым/);

const unknown = familyAccessStatus({});
assert.equal(unknown.key, "unknown");
assert.equal(unknown.label, "Не проверен");

const html = familyStatusPanelHTML([
  { name: "DeepSeek", state: "available", passed_count: 1, total_count: 1 },
  { name: "Anthropic", state: "blocked", passed_count: 0, total_count: 1 },
  { name: "Gemini", state: "partial", passed_count: 1, total_count: 2 },
  { name: "Unknown AI" }
], "example.ru");
assert.match(html, /aria-label="Доступ сайта example\.ru для семейств ИИ-краулеров"/);
assert.match(html, /Лампа горит — доступ открыт/);
assert.match(html, /class="family-status-card is-available"/);
assert.match(html, /class="family-status-card is-blocked"/);
assert.match(html, /class="family-status-card is-partial"/);
assert.match(html, /class="family-status-card is-unknown"/);
assert.match(html, /family-status-lamp-bezel/);
assert.match(html, /family-status-lamp-glass/);
assert.match(html, /family-status-lamp-filament/);
assert.match(html, /\/static\/brand\/providers\/deepseek\.svg/);
assert.match(html, /class="audit-tooltip family-status-tooltip"/);
assert.match(html, /role="tooltip"/);
assert.match(html, /aria-describedby="family-0-access-help"/);
assert.match(html, /Как проверен доступ для DeepSeek/);
assert.doesNotMatch(html, /progress|bar-track|0–100%/i);
''',
            )
        )
        completed = subprocess.run(
            ["node", "-e", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            "Family-status runtime fixture failed:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
