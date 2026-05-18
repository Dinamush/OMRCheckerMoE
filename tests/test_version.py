"""Application version metadata."""

import unittest

import version


class TestVersion(unittest.TestCase):
    def test_version_format(self) -> None:
        parts = version.__version__.split(".")
        self.assertGreaterEqual(len(parts), 2)
        for p in parts:
            self.assertTrue(p.isdigit(), msg=version.__version__)

    def test_version_info_matches(self) -> None:
        self.assertEqual(
            version.__version_info__,
            tuple(int(x) for x in version.__version__.split(".")),
        )


if __name__ == "__main__":
    unittest.main()
