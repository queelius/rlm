from pathlib import Path


def test_commands_preserve_frozen_prefix_source_and_no_new_root_training():
    from launch_next_question import commands

    collect, analyze = commands(Path("/study"))
    assert collect[1] == "/study/source-017/next_question_probe.py"
    assert collect[collect.index("--source-output") + 1] == "/study/helper-transfer-musique-001"
    assert collect[collect.index("--output") + 1] == "/study/next-question-screen-001"
    assert collect[collect.index("--hours") + 1] == "1"
    assert analyze[analyze.index("--report") + 1] == "/study/analysis-next-question-screen-001.json"
