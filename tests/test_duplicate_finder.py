"""Unit tests for duplicate_finder installment / release stripping."""

import unittest

from duplicate_finder import (
    cluster_duplicates,
    normalize_name,
    _is_likely_duplicate_copy,
    _numeric_token_installment_mismatch,
    _same_series_different_installment,
    _series_episode_split,
    _strip_release_metadata,
)


class TestStripRelease(unittest.TestCase):
    def test_strips_resolution_and_encoder_tail(self) -> None:
        self.assertEqual(_strip_release_metadata("foo 720p v1x"), "foo")
        self.assertEqual(_strip_release_metadata("foo 1080p V2X"), "foo")

    def test_empty_after_strip(self) -> None:
        self.assertEqual(_strip_release_metadata("720p v1x"), "")


class TestSeriesEpisodeSplit(unittest.TestCase):
    def test_hyphen_episode_before_release_tail(self) -> None:
        n = normalize_name("title-1-720p-v1x.mp4")
        self.assertEqual(_series_episode_split(n), ("title", (1,)))

    def test_pixiv_page_index(self) -> None:
        n = normalize_name("88071415_p0.png")
        self.assertEqual(_series_episode_split(n), ("88071415", (0,)))

    def test_series_mid_title(self) -> None:
        n = normalize_name("Toshi_Densetsu_Series_05.mp4")
        # Underscores are kept by normalize_name (\w); base matches other files in the set.
        self.assertEqual(_series_episode_split(n), ("toshi_densetsu", (5,)))

    def test_trailing_digit_after_strip(self) -> None:
        n = normalize_name("some kankei 1 720p v1x.mp4")
        self.assertEqual(_series_episode_split(n), ("some kankei", (1,)))

    def test_meru_succubus_sequel_digit(self) -> None:
        n = normalize_name("meru the succubus 3.mp4")
        self.assertEqual(_series_episode_split(n), ("meru the succubus", (3,)))


class TestSameSeriesDifferentInstallment(unittest.TestCase):
    def test_blocks_adjacent_episodes(self) -> None:
        a = normalize_name("title-1-720p.mp4")
        b = normalize_name("title-2-720p.mp4")
        self.assertTrue(_same_series_different_installment(a, b))

    def test_series_episode_numbers(self) -> None:
        a = normalize_name("Toshi_Densetsu_Series_01.mp4")
        b = normalize_name("Toshi_Densetsu_Series_05.mp4")
        self.assertTrue(_same_series_different_installment(a, b))

    def test_pixiv_pages(self) -> None:
        a = normalize_name("91344038_p0.mp4")
        b = normalize_name("91344038_p1.mp4")
        self.assertTrue(_same_series_different_installment(a, b))


class TestLikelyDuplicateCopy(unittest.TestCase):
    def test_windows_copy_same_size(self) -> None:
        a = normalize_name("koinaka-720p-v1x (2).mp4")
        b = normalize_name("koinaka-720p-v1x.mp4")
        self.assertTrue(_is_likely_duplicate_copy(a, b, 234_900_000, 234_900_000))

    def test_episode_suffix_not_copy_when_sizes_differ(self) -> None:
        a = normalize_name("nudist-beach-ni-shuugakuryokou-de-2-720p.mp4")
        b = normalize_name("nudist-beach-ni-shuugakuryokou-de-720p.mp4")
        self.assertFalse(_is_likely_duplicate_copy(a, b, 198_000_000, 167_800_000))


class TestNumericTokenMismatch(unittest.TestCase):
    def test_extra_trailing_episode_token(self) -> None:
        a = normalize_name("nudist-beach-ni-shuugakuryokou-de-2-720p.mp4")
        b = normalize_name("nudist-beach-ni-shuugakuryokou-de-720p.mp4")
        self.assertTrue(_numeric_token_installment_mismatch(a, b))

    def test_hex_like_skipped(self) -> None:
        h1 = "a" * 32
        h2 = ("b" * 31) + "1"
        self.assertFalse(_numeric_token_installment_mismatch(h1, h2))


class TestClusterDuplicates(unittest.TestCase):
    def test_merges_windows_copy_pair(self) -> None:
        files = [
            {"name": "clip 720p v1x.mp4", "size": 10_000_000},
            {"name": "clip 720p v1x (2).mp4", "size": 10_000_000},
        ]
        groups = cluster_duplicates(files, threshold=0.72)
        self.assertEqual(groups, [[0, 1]])

    def test_does_not_merge_different_episodes_same_size(self) -> None:
        files = [
            {"name": "series-1-720p-v1x.mp4", "size": 50_000_000},
            {"name": "series-2-720p-v1x.mp4", "size": 50_000_000},
        ]
        groups = cluster_duplicates(files, threshold=0.72)
        self.assertEqual(groups, [])

    def test_does_not_merge_transitive_bridge_across_episodes(self) -> None:
        files = [
            {"name": "series-1-720p-v1x.mp4", "size": 50_000_000},
            {"name": "series-1-720p-v1x (2).mp4", "size": 50_000_000},
            {"name": "series-2-720p-v1x.mp4", "size": 50_000_000},
        ]
        groups = cluster_duplicates(files, threshold=0.72)
        self.assertEqual(groups, [[0, 1]])


if __name__ == "__main__":
    unittest.main()
