"""M2 DoD: vocabulary identical across processes and PYTHONHASHSEED values."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _vocab_hashes(hashseed: str) -> str:
    env = dict(os.environ, PYTHONHASHSEED=hashseed)
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "hash_vocab.py")],
        capture_output=True,
        text=True,
        check=True,
        env=env,
        cwd=REPO,
    )
    return out.stdout


@pytest.mark.slow
def test_vocabulary_stable_across_pythonhashseed() -> None:
    a = _vocab_hashes("0")
    b = _vocab_hashes("4242")
    assert a == b
    assert "predicate " in a and "predicate_object " in a
