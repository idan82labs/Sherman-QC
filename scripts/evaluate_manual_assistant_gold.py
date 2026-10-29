#!/usr/bin/env python3
"""Gold evaluation for the Sherman manual assistant.

The evaluator calls the same service function used by the API. It scores the
key RAG contract: profile isolation, support-state abstention, manual/page hit
quality, visual evidence availability, and multilingual query handling.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = REPO_ROOT / "data" / "manual_assistant" / "gold_eval_cases.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "tmp" / "manual_assistant_gold_eval"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("SHERMAN_CHAT_PROVIDER", "mock")
os.environ.setdefault("SHERMAN_CHAT_INCLUDE_RETRIEVAL_TRACE", "true")

from apps.api.services.manual_assistant_service import chat as production_chat  # noqa: E402
from apps.api.services.manual_assistant_service import retrieval_chat  # noqa: E402
from domains.manual_assistant.models import ManualAssistantChatRequest  # noqa: E402


SUPPORTED_STATES = {"supported", "partial_support", "partial_support_visual_gap"}


@dataclass(frozen=True)
class GoldCase:
    id: str
    category: str
    language: str
    profile: str
    query: str
    expected_support_state: str
    expected_manual: str | None
    expected_pages: tuple[int, ...] = ()
    requires_visual: bool = False
    forbidden_manuals: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GoldCase":
        return cls(
            id=value["id"],
            category=value["category"],
            language=value.get("language", "en"),
            profile=value["profile"],
            query=value["query"],
            expected_support_state=value["expected_support_state"],
            expected_manual=value.get("expected_manual"),
            expected_pages=tuple(value.get("expected_pages") or ()),
            requires_visual=bool(value.get("requires_visual", False)),
            forbidden_manuals=tuple(value.get("forbidden_manuals") or ()),
        )


@dataclass
class CaseResult:
    case: GoldCase
    endpoint: str
    support_state: str
    response_profile: str
    top_manual: str | None
    top_page: int | None
    top3_pages: list[tuple[str, int]] = field(default_factory=list)
    top5_pages: list[tuple[str, int]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    support_ok: bool = False
    top1_manual_ok: bool = False
    top1_page_ok: bool = False
    top3_page_ok: bool = False
    top5_page_ok: bool = False
    visual_ok: bool = False
    leakage_ok: bool = False
    citation_abstention_ok: bool = False
    trace_abstention_ok: bool = False
    profile_ok: bool = False
    failures: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.case.id,
            "category": self.case.category,
            "language": self.case.language,
            "profile": self.case.profile,
            "query": self.case.query,
            "endpoint": self.endpoint,
            "expected_support_state": self.case.expected_support_state,
            "expected_manual": self.case.expected_manual,
            "expected_pages": list(self.case.expected_pages),
            "requires_visual": self.case.requires_visual,
            "support_state": self.support_state,
            "response_profile": self.response_profile,
            "top_manual": self.top_manual,
            "top_page": self.top_page,
            "top3_pages": self.top3_pages,
            "top5_pages": self.top5_pages,
            "support_ok": self.support_ok,
            "top1_manual_ok": self.top1_manual_ok,
            "top1_page_ok": self.top1_page_ok,
            "top3_page_ok": self.top3_page_ok,
            "top5_page_ok": self.top5_page_ok,
            "visual_ok": self.visual_ok,
            "leakage_ok": self.leakage_ok,
            "citation_abstention_ok": self.citation_abstention_ok,
            "trace_abstention_ok": self.trace_abstention_ok,
            "profile_ok": self.profile_ok,
            "failures": self.failures,
            "warnings": self.warnings,
            "citations": self.citations,
        }


def load_cases(path: Path) -> list[GoldCase]:
    return [GoldCase.from_dict(item) for item in json.loads(path.read_text(encoding="utf-8"))]


def response_pages(items) -> list[tuple[str, int]]:
    return [(item.manual_id, item.page_number) for item in items]


def page_hit(pages: list[tuple[str, int]], expected_manual: str | None, expected_pages: tuple[int, ...]) -> bool:
    if expected_manual is None or not expected_pages:
        return False
    return any(manual == expected_manual and page in expected_pages for manual, page in pages)


def eval_case(case: GoldCase, endpoint: str) -> CaseResult:
    ui_language = "he" if case.language == "he" else "en"
    service = production_chat if endpoint == "chat" else retrieval_chat
    response = service(
        ManualAssistantChatRequest(
            profile=case.profile,  # type: ignore[arg-type]
            message=case.query,
            ui_language=ui_language,
            retrieval_profile="accurate",
        )
    )
    citations = response.citations
    trace = response.retrieval_trace
    top = citations[0] if citations else (trace[0] if trace else None)
    result = CaseResult(
        case=case,
        endpoint=endpoint,
        support_state=response.support_state,
        response_profile=response.profile,
        top_manual=top.manual_id if top else None,
        top_page=top.page_number if top else None,
        top3_pages=response_pages(citations[:3] if citations else trace[:3]),
        top5_pages=response_pages(trace[:5]),
        citations=[
            {
                "manual_id": item.manual_id,
                "page_number": item.page_number,
                "has_crop": item.crop is not None,
                "retrieval": item.retrieval,
            }
            for item in trace[:5]
        ],
        warnings=response.warnings,
    )

    result.profile_ok = response.profile == case.profile

    if case.expected_support_state in {"not_found", "clarification"}:
        result.support_ok = response.support_state == case.expected_support_state
        result.citation_abstention_ok = not response.citations
        result.trace_abstention_ok = not response.retrieval_trace
        result.top1_manual_ok = True
        result.top1_page_ok = True
        result.top3_page_ok = True
        result.top5_page_ok = True
        result.visual_ok = True
    else:
        result.support_ok = response.support_state in SUPPORTED_STATES
        result.citation_abstention_ok = True
        result.trace_abstention_ok = True
        result.top1_manual_ok = bool(citations and citations[0].manual_id == case.expected_manual)
        result.top1_page_ok = page_hit(response_pages(citations[:1]), case.expected_manual, case.expected_pages)
        result.top3_page_ok = page_hit(response_pages(citations[:3]), case.expected_manual, case.expected_pages)
        result.top5_page_ok = page_hit(response_pages(trace[:5]), case.expected_manual, case.expected_pages)
        result.visual_ok = (
            any(item.crop is not None for item in citations[:3])
            if case.requires_visual
            else True
        )

    forbidden = set(case.forbidden_manuals)
    manuals_seen = {manual for manual, _page in result.top5_pages}
    result.leakage_ok = not forbidden.intersection(manuals_seen)

    if not result.support_ok:
        result.failures.append(f"support_state expected {case.expected_support_state}, got {response.support_state}")
    if not result.citation_abstention_ok:
        result.failures.append("not_found response returned citations")
    if not result.trace_abstention_ok:
        result.failures.append("not_found/clarification response returned retrieval_trace")
    if not result.profile_ok:
        result.failures.append(f"response profile changed from {case.profile} to {response.profile}")
    if not result.top1_manual_ok:
        result.failures.append(f"top manual expected {case.expected_manual}, got {result.top_manual}")
    if not result.top5_page_ok:
        result.failures.append(f"expected page {list(case.expected_pages)} not found in top5 trace")
    if not result.visual_ok:
        result.failures.append("visual case did not return a crop citation")
    if not result.leakage_ok:
        result.failures.append(f"forbidden manual appeared in top5 trace: {sorted(forbidden.intersection(manuals_seen))}")
    return result


def pct(values: list[bool]) -> float:
    if not values:
        return 100.0
    return round(sum(1 for value in values if value) * 100 / len(values), 1)


def build_metrics(results: list[CaseResult]) -> dict[str, Any]:
    supported = [
        result for result in results if result.case.expected_support_state not in {"not_found", "clarification"}
    ]
    visual = [result for result in results if result.case.requires_visual]
    multilingual = [result for result in results if result.case.language != "en"]
    negative = [result for result in results if result.case.expected_support_state == "not_found"]
    non_supported_public = [
        result for result in results if result.case.expected_support_state in {"not_found", "clarification"}
    ]

    support_accuracy = pct([result.support_ok for result in results])
    top1_manual_accuracy = pct([result.top1_manual_ok for result in supported])
    top1_page_hit = pct([result.top1_page_ok for result in supported])
    top3_page_hit = pct([result.top3_page_ok for result in supported])
    top5_page_hit = pct([result.top5_page_ok for result in supported])
    visual_evidence = pct([result.visual_ok for result in visual])
    leakage_guard = pct([result.leakage_ok for result in results])
    multilingual_accuracy = pct(
        [
            result.support_ok
            and result.top1_manual_ok
            and result.top5_page_ok
            and (result.visual_ok if result.case.requires_visual else True)
            for result in multilingual
        ]
    )
    abstention_quality = pct([result.support_ok and result.citation_abstention_ok for result in negative])
    trace_abstention_quality = pct([result.support_ok and result.trace_abstention_ok for result in non_supported_public])
    profile_integrity = pct([result.profile_ok for result in results])

    score = (
        1.5 * support_accuracy
        + 1.5 * top1_manual_accuracy
        + 1.5 * top5_page_hit
        + 1.0 * top3_page_hit
        + 1.0 * visual_evidence
        + 1.0 * leakage_guard
        + 1.0 * multilingual_accuracy
        + 0.5 * abstention_quality
        + 0.5 * trace_abstention_quality
        + 0.5 * profile_integrity
    ) / 100

    return {
        "cases": len(results),
        "supported_cases": len(supported),
        "negative_cases": len(negative),
        "visual_cases": len(visual),
        "multilingual_cases": len(multilingual),
        "support_accuracy_pct": support_accuracy,
        "top1_manual_accuracy_pct": top1_manual_accuracy,
        "top1_page_hit_pct": top1_page_hit,
        "top3_page_hit_pct": top3_page_hit,
        "top5_page_hit_pct": top5_page_hit,
        "visual_evidence_pct": visual_evidence,
        "leakage_guard_pct": leakage_guard,
        "multilingual_accuracy_pct": multilingual_accuracy,
        "abstention_quality_pct": abstention_quality,
        "trace_abstention_quality_pct": trace_abstention_quality,
        "profile_integrity_pct": profile_integrity,
        "gold_rag_quality_score_out_of_10": round(score, 2),
    }


def write_report(output_dir: Path, metrics: dict[str, Any], results: list[CaseResult]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manual_assistant_gold_eval.json").write_text(
        json.dumps(
            {
                "metrics": metrics,
                "failures": [result.as_dict() for result in results if result.failures],
                "results": [result.as_dict() for result in results],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Manual Assistant Gold Eval",
        "",
        f"- Cases: {metrics['cases']}",
        f"- Gold RAG quality score: {metrics['gold_rag_quality_score_out_of_10']}/10",
        f"- Support accuracy: {metrics['support_accuracy_pct']}%",
        f"- Top-1 manual accuracy: {metrics['top1_manual_accuracy_pct']}%",
        f"- Top-5 page hit: {metrics['top5_page_hit_pct']}%",
        f"- Visual evidence: {metrics['visual_evidence_pct']}%",
        f"- Multilingual accuracy: {metrics['multilingual_accuracy_pct']}%",
        f"- Leakage guard: {metrics['leakage_guard_pct']}%",
        f"- Abstention quality: {metrics['abstention_quality_pct']}%",
        f"- Trace abstention quality: {metrics['trace_abstention_quality_pct']}%",
        f"- Profile integrity: {metrics['profile_integrity_pct']}%",
        "",
        "## Failures",
        "",
    ]
    failed = [result for result in results if result.failures]
    if not failed:
        lines.append("None.")
    else:
        for result in failed:
            lines.append(
                f"- {result.case.id}: {'; '.join(result.failures)}. "
                f"Endpoint={result.endpoint} Top5={result.top5_pages}"
            )
    (output_dir / "manual_assistant_gold_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-score", type=float, default=9.0)
    parser.add_argument("--endpoint", choices=("retrieval", "chat", "both"), default="both")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    endpoints = ("retrieval", "chat") if args.endpoint == "both" else (args.endpoint,)
    results = [eval_case(case, endpoint) for endpoint in endpoints for case in cases]
    metrics = build_metrics(results)
    write_report(args.output_dir, metrics, results)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    failed = [result for result in results if result.failures]
    if failed:
        print("\nFailures:")
        for result in failed:
            print(f"- {result.case.id} [{result.endpoint}]: {'; '.join(result.failures)} Top5={result.top5_pages}")

    return 0 if metrics["gold_rag_quality_score_out_of_10"] >= args.min_score and not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
