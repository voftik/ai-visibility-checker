import json
import subprocess
import unittest
from pathlib import Path


class FrontendObservationalRuntimeTests(unittest.TestCase):
    """Execute the report helpers as shipped, instead of grepping their source."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.html = Path("static/index.html").read_text(encoding="utf-8")

    def _declaration(self, name: str) -> str:
        marker = f"      const {name} ="
        start = self.html.find(marker)
        self.assertNotEqual(start, -1, f"Missing frontend declaration: {name}")

        next_declaration = self.html.find("\n\n      const ", start + len(marker))
        self.assertNotEqual(
            next_declaration,
            -1,
            f"Could not find the end of frontend declaration: {name}",
        )
        return self.html[start:next_declaration].strip()

    def _run_frontend_fixture(self, assertions: str) -> None:
        declarations = "\n\n".join(
            self._declaration(name)
            for name in (
                "escapeHTML",
                "finiteNumber",
                "safeNumber",
                "formatPercent",
                "firstFinite",
                "metricRate",
                "observationalMemoryLimitation",
                "isObservationalMemoryMetric",
                "memorySlicePresentation",
                "metricPresentation",
                "comparisonMetricCellHTML",
                "comparisonTableHTML",
                "pairedComparisonTableHTML",
                "providerKnowledgeMetric",
            )
        )
        script = "\n".join(
            (
                '"use strict";',
                'const assert = require("node:assert/strict");',
                declarations,
                assertions,
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
            "Frontend runtime fixture failed:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}\n"
            f"script declarations: {json.dumps(declarations[:500])}",
        )

    def test_observational_memory_contract_executes_fail_closed(self) -> None:
        self._run_frontend_fixture(
            r'''
const observationalMemory = {
  data_state: "limited",
  evidence_state: "legacy_observational",
  limitation_reason: "legacy_memory_request_not_enforced",
  strict_no_web_verified: false,
  observational_answers: 6,
  expected_answers: 6,
  valid_answers: 6,
  mention_count: 2,
  mention_rate: 33.3
};
const webMetric = {
  data_state: "complete",
  expected_answers: 5,
  valid_answers: 5,
  mention_count: 4,
  mention_rate: 80
};

const presentation = memorySlicePresentation([observationalMemory]);
assert.equal(presentation.strict, false);
assert.equal(presentation.observational, true);
assert.equal(presentation.columnLabel, "Исторический срез");
assert.match(presentation.caveat, /техническое отключение веба.+не аттестовано/);

const comparison = comparisonTableHTML([{
  label: "Материнский бренд без подсказки",
  web: webMetric,
  memory: observationalMemory,
  options: {
    rateKeys: ["mention_rate"],
    countKey: "mention_count",
    countLabel: "Упоминаний"
  }
}]);
assert.match(comparison, /<span role="columnheader">Исторический срез<\/span>/);
assert.match(comparison, /<strong>33,3%<\/strong>/);
assert.match(comparison, /Упоминаний: 2 из 6\. Исторический срез/);
assert.match(comparison, /class="comparison-observation-note"/);
assert.match(comparison, /техническое отключение веба.+не аттестовано/);
assert.doesNotMatch(comparison, /подтвержд[её]нн(?:ая|ый) память/i);

// Missing provenance must never silently become an attested no-web slice.
const missingProvenance = memorySlicePresentation([{
  data_state: "complete",
  valid_answers: 4,
  mention_count: 2,
  mention_rate: 50
}]);
assert.equal(missingProvenance.strict, false);
assert.equal(missingProvenance.observational, true);
assert.equal(missingProvenance.columnLabel, "Исторический срез");
assert.match(missingProvenance.caveat, /не содержит технической аттестации/);

// The displayed delta is supplied by the server, never recomputed as 90 - 10.
const paired = pairedComparisonTableHTML({
  nPairs: 6,
  rows: [{
    label: "Материнский бренд",
    description: "Проверка серверной разницы.",
    web: {
      data_state: "complete",
      strict_no_web_verified: true,
      valid_answers: 6,
      mention_count: 5,
      mention_rate: 90
    },
    memory: {
      data_state: "complete",
      strict_no_web_verified: true,
      valid_answers: 6,
      mention_count: 1,
      mention_rate: 10
    },
    observedDifference: 7,
    observedDifferenceMetric: "mention_rate_percentage_points"
  }]
});
assert.match(paired, />\+7,0 п\.п\.</);
assert.doesNotMatch(paired, />\+80(?:,0)? п\.п\.</);
assert.match(paired, /не доказанный причинный эффект/);
assert.doesNotMatch(paired, /(?:поиск|веб)[^<.]{0,60}(?:повысил|улучшил|вызвал|обеспечил)/i);

// A bare number without the server metric provenance is not comparable.
const untypedDifference = pairedComparisonTableHTML({
  nPairs: 6,
  causalInterpretationAllowed: false,
  rows: [{
    label: "Материнский бренд",
    description: "Нет типа серверной метрики.",
    web: webMetric,
    memory: observationalMemory,
    observedDifference: 70
  }]
});
assert.match(untypedDifference, />Не сравниваем<\/span>/);
assert.doesNotMatch(untypedDifference, />\+70 п\.п\.</);

// Provider knowledge is memory-only: a web value must not leak into the cell.
const webOnlyProvider = {
  knowledgeMemory: null,
  knowledgeWeb: { data_state: "complete", specific_rate: 77 },
  brand_knowledge: {
    web: { data_state: "complete", specific_rate: 77 }
  }
};
assert.equal(providerKnowledgeMetric(webOnlyProvider), null);
assert.equal(
  providerKnowledgeMetric({ knowledgeMemory: observationalMemory }),
  observationalMemory
);
'''
        )


if __name__ == "__main__":
    unittest.main()
