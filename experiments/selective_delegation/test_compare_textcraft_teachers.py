import compare_textcraft_teachers as teachers


def test_teacher_contrast_pairs_profiles_and_preserves_missing():
    jobs = [
        dict(episode_id=f"{p}{r}", task_id=p, repeat=r, prompt_profile="original")
        for p in ("a", "b")
        for r in (0, 1)
    ]
    left = {j["episode_id"]: {**j, "observed": True, "native_score": 0} for j in jobs}
    right = {k: {**v, "native_score": int(v["task_id"] == "a")} for k, v in left.items()}
    result = teachers.contrasts(jobs, left, right)["original"]
    assert result["wins"] == 2 and result["complete_panel_difference"] == 0.5
    assert result["planned_pairs"] == 4
    del right["a0"]
    result = teachers.contrasts(jobs, left, right)["original"]
    assert result["unknown_pairs"] == 1 and result["complete_panel_difference"] is None
    assert result["ci95"] is None
