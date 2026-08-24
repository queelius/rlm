import pytest

from rlm.json import StrictJSONError, strict_json_loads


@pytest.mark.parametrize("text", ['{"key": 1, "key": 2}', "NaN", "Infinity"])
def test_strict_json_rejects_non_json_decoder_extensions(text: str) -> None:
    with pytest.raises(StrictJSONError):
        strict_json_loads(text)
