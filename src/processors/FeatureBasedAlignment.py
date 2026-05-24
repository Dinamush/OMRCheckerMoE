"""
Image based feature alignment
Credits: https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
"""
import cv2
import numpy as np

from src.logger import logger
from src.processors.interfaces.ImagePreprocessor import ImagePreprocessor
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils
from src.constants.image_processing import (
    DEFAULT_MAX_FEATURES,
    DEFAULT_GOOD_MATCH_PERCENT
)

# Minimum point correspondences required by cv2.findHomography
_MIN_HOMOGRAPHY_POINTS = 4


class FeatureBasedAlignment(ImagePreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        options = self.options
        config = self.tuning_config

        # process reference image
        self.ref_path = self.relative_dir.joinpath(options["reference"])
        ref_img = cv2.imread(str(self.ref_path), cv2.IMREAD_GRAYSCALE)
        # Audit fix CORE-6: cv2.imread returns None on missing / corrupted /
        # unsupported reference images. Without this guard ImageUtils.resize_util
        # immediately raises a cryptic AttributeError on `.shape`.
        if ref_img is None:
            raise FileNotFoundError(
                f"FeatureBasedAlignment reference image not found or unreadable: "
                f"{self.ref_path}"
            )
        self.ref_img = ImageUtils.resize_util(
            ref_img,
            config.dimensions.processing_width,
            config.dimensions.processing_height,
        )
        # get options with defaults
        self.max_features = int(options.get("maxFeatures", DEFAULT_MAX_FEATURES))
        self.good_match_percent = options.get("goodMatchPercent", DEFAULT_GOOD_MATCH_PERCENT)
        self.transform_2_d = options.get("2d", False)
        # Extract keypoints and description of source image
        self.orb = cv2.ORB_create(self.max_features)
        self.to_keypoints, self.to_descriptors = self.orb.detectAndCompute(
            self.ref_img, None
        )

    def __str__(self):
        return self.ref_path.name

    def exclude_files(self):
        return [self.ref_path]

    def apply_filter(self, image, _file_path):
        config = self.tuning_config

        # NOTE: like ImageUtils.normalize_util, this call relies on the
        # alpha-in-dst-slot pattern that effectively inverts the image. The
        # downstream pipeline expects this inversion. See the long-form
        # comment in src/utils/image.py:normalize_util for context.
        image = cv2.normalize(image, 0, 255, norm_type=cv2.NORM_MINMAX)

        # Detect ORB features and compute descriptors.
        from_keypoints, from_descriptors = self.orb.detectAndCompute(image, None)

        # Audit fix CORE-8: a blank / over-exposed / very low-contrast scan
        # can return zero ORB descriptors. Passing None / empty to
        # matcher.match() raises cv2.error. Same for the reference side.
        if (
            from_descriptors is None
            or len(from_descriptors) == 0
            or self.to_descriptors is None
            or len(self.to_descriptors) == 0
        ):
            logger.warning(
                "FeatureBasedAlignment: no ORB descriptors on input or reference, "
                "skipping alignment for this image."
            )
            return image

        # Match features.
        matcher = cv2.DescriptorMatcher_create(
            cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
        )

        matches = np.array(matcher.match(from_descriptors, self.to_descriptors, None))

        # Sort matches by score
        matches = sorted(matches, key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        num_good_matches = int(len(matches) * self.good_match_percent)
        matches = matches[:num_good_matches]

        # Audit fix CORE-14: cv2.findHomography needs at least 4 point
        # correspondences. Heavily JPEG-compressed or near-blank scans
        # produce fewer good matches and previously crashed downstream.
        if len(matches) < _MIN_HOMOGRAPHY_POINTS:
            logger.warning(
                "FeatureBasedAlignment: only %d good matches (need %d); "
                "skipping alignment.",
                len(matches),
                _MIN_HOMOGRAPHY_POINTS,
            )
            return image

        # Draw top matches
        if config.outputs.show_image_level > 2:
            im_matches = cv2.drawMatches(
                image, from_keypoints, self.ref_img, self.to_keypoints, matches, None
            )
            InteractionUtils.show("Aligning", im_matches, resize=True, config=config)

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = from_keypoints[match.queryIdx].pt
            points2[i, :] = self.to_keypoints[match.trainIdx].pt

        # Find homography
        height, width = self.ref_img.shape

        # Audit fix CORE-7: cv2.estimateAffine2D / cv2.findHomography return
        # None when RANSAC cannot find enough inliers. Passing None to
        # warpAffine / warpPerspective raised an opaque cv2.error inside the
        # worker. Fall back to passthrough so the image continues through
        # the pipeline and is later judged by downstream stages.
        if self.transform_2_d:
            m, _inliers = cv2.estimateAffine2D(points1, points2)
            if m is None:
                logger.warning(
                    "FeatureBasedAlignment: estimateAffine2D returned None "
                    "(insufficient inliers); skipping alignment."
                )
                return image
            return cv2.warpAffine(image, m, (width, height))

        h, _mask = cv2.findHomography(points1, points2, cv2.RANSAC)
        if h is None:
            logger.warning(
                "FeatureBasedAlignment: findHomography returned None "
                "(insufficient RANSAC inliers); skipping alignment."
            )
            return image
        return cv2.warpPerspective(image, h, (width, height))
