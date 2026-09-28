"""Exact local-source adapter rejects an unexpected native client seam."""

import unittest

import native_adapter


def example():
    return 1


class NativeBindingTests(unittest.TestCase):
    def test_counted_rewrite_preserves_an_executable_function(self):
        function, receipt = native_adapter.rewrite(example, {}, [("return 1", "return 2")])
        self.assertEqual(function(), 2)
        self.assertNotEqual(receipt["original_sha256"], receipt["transformed_sha256"])

    def test_missing_source_seam_is_rejected(self):
        with self.assertRaises(ValueError):
            native_adapter.rewrite(example, {}, [("return 3", "return 2")])


if __name__ == "__main__":
    unittest.main()
