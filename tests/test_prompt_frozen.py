from __future__ import annotations

import hashlib

from trading_copilot.analysis.prompting import ANALYSIS_SYSTEM_PROMPT, PROMPT_VERSION


def test_approved_prompt_v04_is_frozen() -> None:
    assert PROMPT_VERSION == "0.4"
    assert (
        hashlib.sha256(ANALYSIS_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
        == "f276de3ae7fcbcab046775fe0729bfa2fc309c2db4265ddad21b22c8b59e825a"
    )
