"""Checks that presenter notes cannot silently attach to the wrong slides."""

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path


class PresenterTests(unittest.TestCase):
    def test_default_talk_timer_is_ten_minutes(self):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("present.py")), "--dry-run"],
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertIn("--duration=10", result.stdout)

    def setUp(self):
        path = Path(__file__).with_name("present.py")
        self.assertTrue(path.exists(), "The presenter launcher has not been implemented")
        spec = importlib.util.spec_from_file_location("present", path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_matches_notes_to_titles_and_pdfpc_zero_based_pages(self):
        source = r"\frametitle{First\\idea}\frametitle{Second idea}"
        notes = {
            "slides": [
                {"title": "First idea", "notes": "Say this."},
                {"title": "Second idea", "notes": "Then explain that."},
            ]
        }
        result = self.module.make_metadata(source, notes)
        self.assertEqual(result["pdfpcFormat"], 2)
        self.assertTrue(result["disableMarkdown"])
        self.assertEqual(
            result["pages"],
            [
                {"idx": 0, "label": "1", "overlay": 0, "note": "Say this."},
                {"idx": 1, "label": "2", "overlay": 0, "note": "Then explain that."},
            ],
        )

    def test_rejects_reordered_or_missing_notes(self):
        source = r"\frametitle{First idea}\frametitle{Second idea}"
        for slides in [
            [],
            [
                {"title": "Second idea", "notes": "Wrong page."},
                {"title": "First idea", "notes": "Wrong page."},
            ],
        ]:
            with self.subTest(slides=slides), self.assertRaises(ValueError):
                self.module.make_metadata(source, {"slides": slides})

    def test_rejects_empty_or_overlong_notes(self):
        for note in ["", "word " * 101, "x" * 851]:
            with self.subTest(note=note), self.assertRaises(ValueError):
                self.module.make_metadata(
                    r"\frametitle{One}", {"slides": [{"title": "One", "notes": note}]}
                )

    def test_plain_notes_wrap_instead_of_clipping_in_pdfpc46(self):
        note = "SAY: " + "A useful explanation. " * 6 + "\n\nCAUTION: This is preliminary."
        result = self.module.make_metadata(
            r"\frametitle{One}", {"slides": [{"title": "One", "notes": note}]}
        )
        output = result["pages"][0]["note"]
        self.assertTrue(all(len(line) <= 46 for line in output.splitlines()))
        self.assertEqual(" ".join(output.split()), " ".join(note.split()))
        self.assertIn("\n\nCAUTION:", output)

    def test_rejects_notes_taller_than_the_laptop_pane(self):
        note = "\n\n".join(["A brief cue."] * 10)
        with self.assertRaises(ValueError):
            self.module.make_metadata(
                r"\frametitle{One}", {"slides": [{"title": "One", "notes": note}]}
            )

    def test_laptop_opens_both_windows_not_presenter_only(self):
        args = self.module.launch_args("pdfpc", "talk.pdf", "notes.pdfpc", "laptop", 15)
        self.assertIn("--windowed=both", args)
        self.assertNotIn("--single-screen", args)
        self.assertIn("--note-format=plain", args)
        self.assertIn("--pdfpc-location=notes.pdfpc", args)
        self.assertEqual(args[-1], "talk.pdf")

    def test_rehearsal_shows_presenter_on_one_screen(self):
        args = self.module.launch_args("pdfpc", "talk.pdf", "notes.pdfpc", "rehearse", 15)
        self.assertIn("--single-screen", args)
        self.assertIn("--windowed=none", args)
        self.assertNotIn("--switch-screens", args)

    def test_dual_display_does_not_force_single_screen(self):
        args = self.module.launch_args("pdfpc", "talk.pdf", "notes.pdfpc", "dual", 15)
        self.assertNotIn("--single-screen", args)
        self.assertIn("--windowed=none", args)


if __name__ == "__main__":
    unittest.main()
