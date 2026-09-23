from analyze_textcraft_quantity_matched import public_minus_corrected


class FakeProfiles:
    @staticmethod
    def compare(jobs, left, right):
        return {"delta": right["score"] - left["score"], "jobs": jobs}


def test_public_minus_corrected_uses_right_minus_left_profile_semantics() -> None:
    result = public_minus_corrected(FakeProfiles, [{"task_id": "a"}], {"score": 1}, {"score": 0})
    assert result["delta"] == 1
