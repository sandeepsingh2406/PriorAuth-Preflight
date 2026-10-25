"""Tests for eval/metrics.py: the SPEC section 2 measures and the results report."""
import pytest

from eval.metrics import (
    CaseResult,
    Measures,
    compute_measures,
    evaluate_case,
    group_of,
    measure_range,
    meets_target,
    render_report,
    results_to_json,
)
from src.agents import AnalysisRun
from src.models import ExtractedField, Gap, GapReport

NOTE = "SYNTHETIC NOTE\nStaging: Stage II.\nLabs: CBC drawn 2026-04-22.\n"


def _r(
    case_id: str = "gen_001",
    expected: str = "NOT_READY",
    actual: str = "NOT_READY",
    hits: set | None = None,
    missed: set | None = None,
    extra: set | None = None,
    *,
    ambiguous: bool = False,
    raw_failures: int = 0,
    raw_total: int = 5,
    final_total: int = 4,
    final_invalid: int = 0,
) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        expected_status=expected,
        actual_status=actual,
        hits=hits or set(),
        missed=missed or set(),
        extra=extra or set(),
        raw_quote_failures=[f"q{i}" for i in range(raw_failures)],
        raw_quotes_total=raw_total,
        final_quotes_verbatim=final_invalid == 0,
        ambiguous=ambiguous,
        final_quotes_total=final_total,
        final_quotes_invalid=final_invalid,
    )


STAGE = ("stage", "missing")
LAB = ("recent_lab", "stale")
PLAN = ("treatment_plan", "conflicting")


def test_gap_recall_is_hits_over_expected_and_ignores_ambiguous_cases() -> None:
    """3 of 4 expected gaps found = 0.75; an ambiguous case's gaps are not counted."""
    results = [
        _r("gen_001", hits={STAGE, LAB}, missed={("prior_therapies", "missing")}),
        _r("gen_002", hits={LAB}),
        _r("gen_003", "NEEDS_REVIEW", "NEEDS_REVIEW", missed={PLAN}, ambiguous=True),
    ]
    assert compute_measures(results).gap_recall == pytest.approx(3 / 4)


def test_false_alarm_rate_is_extra_over_flagged_and_ignores_ambiguous_cases() -> None:
    """1 of 4 flagged gaps is not real = 0.25."""
    results = [
        _r("gen_001", hits={STAGE, LAB}, extra={PLAN}),
        _r("gen_002", hits={LAB}),
        _r("gen_003", "NEEDS_REVIEW", "NEEDS_REVIEW", extra={STAGE}, ambiguous=True),
    ]
    assert compute_measures(results).false_alarm_rate == pytest.approx(1 / 4)


def test_undefined_rates_are_none_not_zero() -> None:
    """With nothing expected, or nothing flagged, the rate does not exist."""
    m = compute_measures([_r("gen_001", "READY", "READY")])
    assert m.gap_recall is None
    assert m.false_alarm_rate is None
    assert m.escalation_correctness is None


def test_escalation_correctness_counts_ambiguous_cases_routed_to_review() -> None:
    results = [
        _r("gen_001", "NEEDS_REVIEW", "NEEDS_REVIEW", ambiguous=True),
        _r("gen_002", "NEEDS_REVIEW", "NEEDS_REVIEW", ambiguous=True),
        _r("gen_003", "NEEDS_REVIEW", "NOT_READY", ambiguous=True),
    ]
    assert compute_measures(results).escalation_correctness == pytest.approx(2 / 3)


def test_false_escalations_are_non_ambiguous_cases_sent_to_review() -> None:
    results = [
        _r("gen_001", "READY", "NEEDS_REVIEW"),
        _r("gen_002", "READY", "READY"),
        _r("gen_003", "NEEDS_REVIEW", "NEEDS_REVIEW", ambiguous=True),
    ]
    assert compute_measures(results).false_escalation_rate == pytest.approx(1 / 2)


def test_status_accuracy() -> None:
    results = [_r("gen_001"), _r("gen_002", "READY", "NOT_READY"), _r("gen_003"), _r("gen_004")]
    assert compute_measures(results).status_accuracy == pytest.approx(3 / 4)


def test_raw_citation_validity_counts_model_quotes_that_failed() -> None:
    """The model's own quotes: 2 failures out of 10 claimed = 0.8."""
    results = [_r("gen_001", raw_failures=2, raw_total=6), _r("gen_002", raw_total=4)]
    assert compute_measures(results).raw_citation_validity == pytest.approx(8 / 10)


def test_reported_citation_validity_counts_quotes_in_final_reports() -> None:
    """The quotes a person would see: all verbatim = 1.0, one bad = below 1."""
    ok = [_r("gen_001", final_total=3), _r("gen_002", final_total=2)]
    assert compute_measures(ok).reported_citation_validity == 1.0
    bad = ok + [_r("gen_003", final_total=5, final_invalid=1)]
    assert compute_measures(bad).reported_citation_validity == pytest.approx(9 / 10)


def test_groups_are_told_apart_by_the_case_id_prefix() -> None:
    assert group_of("hand_03") == "hand"
    assert group_of("gen_030") == "generated"
    with pytest.raises(ValueError):
        group_of("other_1")


def test_range_across_runs_skips_undefined_values() -> None:
    assert measure_range([0.8, 0.9, 1.0]) == (0.8, 1.0)
    assert measure_range([None, 0.5, None]) == (0.5, 0.5)
    assert measure_range([None, None]) is None


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("gap_recall", 0.90, True),
        ("gap_recall", 0.89, False),
        ("false_alarm_rate", 0.10, True),
        ("false_alarm_rate", 0.11, False),
        ("reported_citation_validity", 1.0, True),
        ("reported_citation_validity", 0.99, False),
        ("escalation_correctness", 0.80, True),
        ("escalation_correctness", 0.79, False),
        ("status_accuracy", 0.1, None),  # no target in SPEC section 2
        ("gap_recall", None, None),
    ],
)
def test_targets_come_from_spec_section_2(name: str, value: float | None, expected: bool | None) -> None:
    assert meets_target(name, value) is expected


# --- per-case evaluation ----------------------------------------------------


def _run(status: str, gaps: list[tuple[str, str]], raw: list[ExtractedField], final: list[ExtractedField]) -> AnalysisRun:
    report = GapReport(
        case_id="gen_001",
        payer_id="payer_a",
        status=status,  # type: ignore[arg-type]
        fields=final,
        gaps=[Gap(item=i, reason=r, detail="d") for i, r in gaps],  # type: ignore[arg-type]
    )
    return AnalysisRun(report=report, raw_fields=raw, tools_called=[])


def _f(name: str, quote: str | None, conflicting: list[str] | None = None) -> ExtractedField:
    return ExtractedField(name=name, value=quote, quote=quote, found=quote is not None, conflicting_quotes=conflicting or [])


def test_evaluate_case_records_ambiguity_and_counts_final_quotes() -> None:
    final = [_f("stage", "Staging: Stage II.", ["CBC drawn 2026-04-22"]), _f("biomarker", None)]
    label = {"expected_status": "NEEDS_REVIEW", "expected_gaps": [{"item": "stage", "reason": "conflicting"}], "ambiguous": True}
    result = evaluate_case("gen_001", label, NOTE, _run("NEEDS_REVIEW", [("stage", "conflicting")], final, final))
    assert result.ambiguous is True
    assert result.hits == {("stage", "conflicting")}
    assert result.final_quotes_total == 2 and result.final_quotes_invalid == 0


def test_evaluate_case_flags_raw_and_final_quote_problems() -> None:
    raw = [_f("stage", "Staging: Stage II."), _f("biomarker", "EGFR positive")]
    final = [_f("stage", "Staging: Stage II."), _f("biomarker", "EGFR positive")]  # a bug if this ever happened
    label = {"expected_status": "NOT_READY", "expected_gaps": [], "ambiguous": False}
    result = evaluate_case("gen_001", label, NOTE, _run("NOT_READY", [], raw, final))
    assert result.raw_quote_failures == ["biomarker: 'EGFR positive'"]
    assert result.final_quotes_invalid == 1 and result.final_quotes_verbatim is False


# --- report -----------------------------------------------------------------


def _data(runs: list[list[CaseResult]], model: str = "test-model") -> dict:
    return results_to_json(model, runs)


def test_json_has_measures_per_group_and_every_case() -> None:
    data = _data([[_r("hand_01", "READY", "READY"), _r("gen_001", hits={STAGE})]])
    (run,) = data["runs"]
    assert data["model"] == "test-model"
    assert set(run["measures"]) == {"hand", "generated"}
    assert [c["case_id"] for c in run["cases"]] == ["hand_01", "gen_001"]
    assert run["measures"]["generated"]["gap_recall"] == 1.0


def test_report_shows_both_groups_targets_ranges_and_the_safety_line() -> None:
    run1 = [_r("hand_01", hits={STAGE}), _r("gen_001", hits={STAGE}, missed={LAB})]
    run2 = [_r("hand_01", hits={STAGE}), _r("gen_001", hits={STAGE, LAB})]
    text = render_report(_data([run1, run2]))
    assert "test-model" in text
    assert "Hand-labeled cases" in text and "Generated cases" in text
    assert ">= 0.90" in text and "<= 0.10" in text and "0.80" in text
    assert "0.50" in text and "1.00" in text  # generated recall: run 1 = 1/2, run 2 = 2/2
    assert "Automatic submissions or sends" in text and "| 0 |" in text


def test_report_lists_each_failing_case_with_its_runs_and_details() -> None:
    run1 = [_r("gen_007", "NEEDS_REVIEW", "NOT_READY", ambiguous=True)]
    run2 = [_r("gen_007", "NEEDS_REVIEW", "NEEDS_REVIEW", ambiguous=True)]
    run3 = [_r("gen_007", "NEEDS_REVIEW", "NOT_READY", ambiguous=True)]
    text = render_report(_data([run1, run2, run3]))
    assert "gen_007" in text
    assert "runs 1, 3" in text
    assert "expected NEEDS_REVIEW, got NOT_READY" in text


def test_report_with_no_failures_says_so() -> None:
    text = render_report(_data([[_r("hand_01", "READY", "READY")]]))
    assert "No failures" in text


def test_report_compares_with_a_baseline_and_shows_what_changed() -> None:
    base = _data([[_r("gen_001", hits={STAGE}, missed={LAB})]], model="base-model")
    final = _data([[_r("gen_001", hits={STAGE, LAB})]])
    text = render_report(final, baseline=base, changes="Added unreadable-passage rule.")
    assert "Baseline" in text and "base-model" in text
    assert "0.50" in text and "1.00" in text
    assert "Added unreadable-passage rule." in text


def test_measures_dataclass_serialises_to_plain_numbers() -> None:
    m = compute_measures([_r("gen_001", hits={STAGE})])
    assert isinstance(m, Measures)
    assert m.cases == 1


def test_unreadable_passages_count_as_quotes_in_the_evaluation() -> None:
    """The model's reported passages are checked like quotes: raw claims and final report."""
    label = {"expected_status": "NEEDS_REVIEW", "expected_gaps": [], "ambiguous": True}
    bad = "EG?R ex@n [unreadable]"
    note = NOTE + bad + "\n"
    report = GapReport(
        case_id="gen_005", payer_id="payer_a", status="NEEDS_REVIEW", fields=[], gaps=[], unreadable_passages=[bad]
    )
    run = AnalysisRun(report=report, raw_fields=[], tools_called=[], raw_unreadable=[bad, "invented passage"])
    result = evaluate_case("gen_005", label, note, run)
    assert result.raw_quotes_total == 2
    assert result.raw_quote_failures == ["unreadable: 'invented passage'"]
    assert result.final_quotes_total == 1 and result.final_quotes_invalid == 0
