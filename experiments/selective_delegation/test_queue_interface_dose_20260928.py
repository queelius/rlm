"""Exercise real saved plan contracts without loading a model."""

import queue_interface_dose_20260928 as queue


def test_three_trainings_and_ten_fixed_readouts_have_exact_source_pins():
    jobs = queue.make_jobs()
    assert len(jobs) == 23
    assert sum(bool(job["output"]) for job in jobs) == 13
    assert sum("cp46-" in job["name"] and job["output"] is not None for job in jobs) == 4
    assert sum("cp69-" in job["name"] and job["output"] is not None for job in jobs) == 4
    for filename, expected in jobs[0]["pins"].items():
        assert queue.digest(queue.Path(filename)) == expected
