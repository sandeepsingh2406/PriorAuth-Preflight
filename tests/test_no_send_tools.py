"""Safety invariant: nothing in src/ can submit, email, fax, or send anything.

SPEC section 2 requires zero automatic submissions or sends, "no such tool exists".
This test scans every module in src/ so a future file is covered too.
"""
import ast
import re
from pathlib import Path

import pytest

SRC = Path(__file__).parent.parent / "src"
SENDING_WORDS = re.compile(r"(^|_)(send|submit|email|fax|post|smtp|upload|publish|notify)(_|$)", re.I)
# Modules whose job is to move data out of the process.
OUTBOUND_MODULES = {
    "smtplib", "ftplib", "imaplib", "poplib", "socket", "requests", "httpx", "aiohttp", "urllib",
}


def _modules() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def test_there_are_modules_to_scan() -> None:
    """Guards the scan itself: an empty list would make every check below pass for free."""
    names = {p.name for p in _modules()}
    assert {"models.py", "tools.py"} <= names


@pytest.mark.parametrize("module", _modules(), ids=lambda p: p.name)
def test_no_function_is_named_like_a_sending_action(module: Path) -> None:
    """No function or method is called send_*, submit_*, email_*, post_*, and so on."""
    tree = ast.parse(module.read_text())
    names = [
        n.name
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    assert [n for n in names if SENDING_WORDS.search(n)] == []


@pytest.mark.parametrize("module", _modules(), ids=lambda p: p.name)
def test_no_outbound_network_or_mail_imports(module: Path) -> None:
    """Our own code imports no mail, socket or HTTP client module."""
    tree = ast.parse(module.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported & OUTBOUND_MODULES == set()


def test_the_scan_would_catch_a_violation() -> None:
    """Self-check: the patterns flag obvious offenders and allow the real tool names."""
    for bad in ["send_email", "submit_packet", "post_to_payer", "fax", "notify_team"]:
        assert SENDING_WORDS.search(bad), bad
    for ok in ["write_audit_log", "escalate_to_human", "check_packet", "validate_citations"]:
        assert not SENDING_WORDS.search(ok), ok
