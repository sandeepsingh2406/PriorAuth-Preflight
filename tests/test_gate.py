"""Tests for the gate's comparison and verdict logic (no model calls)."""
from scripts.gate import DEMO_CASES, CaseResult, evaluate_case, gate_verdict
from src.agents import AnalysisRun
from src.models import ExtractedField, Gap, GapReport

NOTE = "SYNTHETIC NOTE\nStaging: Stage II.\nLabs: CBC drawn 2026-04-22.\n"


def _run(status: str, gaps: list[tuple[str, str]], raw: list[ExtractedField]) -> AnalysisRun:
    report = GapReport(
        case_id="hand_01",
        payer_id="payer_a",
        status=status,  # type: ignore[arg-type]
        fields=[f for f in raw if f.quote is None or f.quote in NOTE],
        gaps=[Gap(item=i, reason=r, detail="d") for i, r in gaps],  # type: ignore[arg-type]
    )
    return AnalysisRun(report=report, raw_fields=raw, tools_called=[])


def _f(name: str, quote: str | None, conflicting: list[str] | None = None) -> ExtractedField:
    return ExtractedField(
        name=name, value=quote, quote=quote, found=quote is not None,
        conflicting_quotes=conflicting or [],
    )


LABEL = {
    "expected_status": "NOT_READY",
    "expected_gaps": [{"item": "biomarker", "reason": "missing"}, {"item": "stage", "reason": "missing"}],
}


def test_evaluate_counts_hits_misses_and_extras() -> None:
    run = _run("NOT_READY", [("biomarker", "missing"), ("recent_lab", "stale")], [])
    result = evaluate_case("hand_01", LABEL, NOTE, run)
    assert result.status_match is True
    assert result.hits == {("biomarker", "missing")}
    assert result.missed == {("stage", "missing")}
    assert result.extra == {("recent_lab", "stale")}


def test_evaluate_flags_raw_quotes_that_fail_validation() -> None:
    """A quote the model claimed but that is not in the note is a citation failure."""
    raw = [
        _f("stage", "Staging: Stage II."),
        _f("biomarker", "EGFR positive"),
        _f("recent_lab", "Labs: CBC drawn 2026-04-22.", conflicting=["CBC 2026-05-01"]),
        _f("diagnosis", None),
    ]
    result = evaluate_case("hand_01", LABEL, NOTE, _run("NOT_READY", [], raw))
    assert result.raw_quote_failures == ["biomarker: 'EGFR positive'", "recent_lab: 'CBC 2026-05-01'"]
    assert result.raw_quotes_total == 3 + 1


def test_final_report_quotes_are_checked_too() -> None:
    raw = [_f("stage", "Staging: Stage II.")]
    assert evaluate_case("hand_01", LABEL, NOTE, _run("NOT_READY", [], raw)).final_quotes_verbatim


def _result(case_id: str, match: bool, *, gaps_ok: bool = True, bad_quotes: int = 0) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        expected_status="READY",
        actual_status="READY" if match else "NOT_READY",
        hits=set(),
        missed=set() if gaps_ok else {("stage", "missing")},
        extra=set(),
        raw_quote_failures=[f"x{i}" for i in range(bad_quotes)],
        raw_quotes_total=5,
        final_quotes_verbatim=True,
    )


def _ten(mismatches: set[str] = frozenset(), **kw: object) -> list[CaseResult]:
    return [_result(f"hand_{i:02d}", f"hand_{i:02d}" not in mismatches, **kw) for i in range(1, 11)]


def test_gate_passes_with_eight_of_ten_and_correct_demo_cases() -> None:
    verdict = gate_verdict(_ten({"hand_02", "hand_03"}), audit_ok=True)
    assert verdict.passed, verdict.failures


def test_gate_fails_below_eight_of_ten() -> None:
    verdict = gate_verdict(_ten({"hand_02", "hand_03", "hand_04"}), audit_ok=True)
    assert not verdict.passed
    assert any("7/10" in f for f in verdict.failures)


def test_gate_fails_if_a_demo_case_is_wrong() -> None:
    demo = sorted(DEMO_CASES)[0]
    verdict = gate_verdict(_ten({demo}), audit_ok=True)
    assert not verdict.passed
    assert any(demo in f for f in verdict.failures)


def test_gate_fails_if_a_demo_case_has_the_wrong_gaps() -> None:
    results = _ten()
    demo = sorted(DEMO_CASES)[0]
    results = [_result(r.case_id, True, gaps_ok=r.case_id != demo) for r in results]
    assert not gate_verdict(results, audit_ok=True).passed


def test_gate_fails_on_any_raw_citation_failure() -> None:
    results = _ten()
    results[4] = _result("hand_05", True, bad_quotes=1)
    verdict = gate_verdict(results, audit_ok=True)
    assert not verdict.passed
    assert any("citation" in f for f in verdict.failures)


def test_audit_problem_is_reported_but_is_not_a_gate_condition() -> None:
    """You set three gate conditions; the audit check is reported alongside, not added to them."""
    verdict = gate_verdict(_ten(), audit_ok=False)
    assert verdict.passed
    assert verdict.notes
