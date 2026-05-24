"""

 OMRChecker

 Author: Udayraj Deshmukh
 Github: https://github.com/Udayraj123

"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.logger import logger

plt.rcParams["figure.figsize"] = (10.0, 8.0)
CLAHE_HELPER = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))


class ImageUtils:
    """A Static-only Class to hold common image processing utilities & wrappers over OpenCV functions"""

    @staticmethod
    def save_img(path, final_marked):
        logger.info(f"Saving Image to '{path}'")
        cv2.imwrite(path, final_marked)

    @staticmethod
    def resize_util(img, u_width, u_height=None):
        # Audit fix CORE-12 (companion): guard against zero-dimension input
        # and target sizes that would otherwise raise an opaque
        # cv2.error("dsize.width and dsize.height are both 0").
        if img is None:
            raise ValueError("resize_util: input image is None")
        if u_height is None:
            h, w = img.shape[:2]
            if w == 0 or h == 0:
                raise ValueError(
                    f"resize_util: cannot compute aspect for zero-sized image {img.shape}"
                )
            u_height = int(h * u_width / w)
        u_width_i = max(1, int(u_width))
        u_height_i = max(1, int(u_height))
        return cv2.resize(img, (u_width_i, u_height_i))

    @staticmethod
    def resize_util_h(img, u_height, u_width=None):
        if img is None:
            raise ValueError("resize_util_h: input image is None")
        if u_width is None:
            h, w = img.shape[:2]
            if w == 0 or h == 0:
                raise ValueError(
                    f"resize_util_h: cannot compute aspect for zero-sized image {img.shape}"
                )
            u_width = int(w * u_height / h)
        u_width_i = max(1, int(u_width))
        u_height_i = max(1, int(u_height))
        return cv2.resize(img, (u_width_i, u_height_i))

    @staticmethod
    def grab_contours(cnts):
        # source: imutils package

        # if the length the contours tuple returned by cv2.findContours
        # is '2' then we are using either OpenCV v2.4, v4-beta, or
        # v4-official
        if len(cnts) == 2:
            cnts = cnts[0]

        # if the length of the contours tuple is '3' then we are using
        # either OpenCV v3, v4-pre, or v4-alpha
        elif len(cnts) == 3:
            cnts = cnts[1]

        # otherwise OpenCV has changed their cv2.findContours return
        # signature yet again and I have no idea WTH is going on
        else:
            raise Exception(
                (
                    "Contours tuple must have length 2 or 3, "
                    "otherwise OpenCV changed their cv2.findContours return "
                    "signature yet again. Refer to OpenCV's documentation "
                    "in that case"
                )
            )

        # return the actual contours array
        return cnts

    @staticmethod
    def normalize_util(img, alpha=0, beta=255):
        # NOTE (audit finding CORE-13, intentionally NOT changed):
        # The positional call below is unusual — ``alpha`` (the variable, 0)
        # lands in cv2.normalize's ``dst`` slot and ``beta`` (255) lands in
        # cv2.normalize's ``alpha`` slot. With ``dst=0`` OpenCV allocates a
        # fresh output and uses alpha=255, beta=0 for the NORM_MINMAX rescale
        # — effectively inverting the image. The rest of this pipeline
        # (threshold direction, morphology, "looks like a Xeroxed OMR" log
        # message, integral-image bubble scoring) all assume this inverted
        # representation, and every snapshot test is recorded against it.
        # Rewriting it to the textbook ``cv2.normalize(img, None, alpha=0,
        # beta=255, norm_type=cv2.NORM_MINMAX)`` changes pixel polarity and
        # breaks the entire downstream chain. Left as-is; the audit fix is
        # the comment above so a future contributor doesn't innocently
        # "tidy" this back into a real bug.
        return cv2.normalize(img, alpha, beta, norm_type=cv2.NORM_MINMAX)

    @staticmethod
    def auto_canny(image, sigma=0.93):
        # compute the median of the single channel pixel intensities
        v = np.median(image)

        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)

        # return the edged image
        return edged

    @staticmethod
    def adjust_gamma(image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")

        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

    @staticmethod
    def four_point_transform(image, pts):
        # obtain a consistent order of the points and unpack them
        # individually
        rect = ImageUtils.order_points(pts)
        (tl, tr, br, bl) = rect

        # compute the width of the new image, which will be the
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

        max_width = max(int(width_a), int(width_b))
        # max_width = max(int(np.linalg.norm(br-bl)), int(np.linalg.norm(tr-tl)))

        # compute the height of the new image, which will be the
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))
        # max_height = max(int(np.linalg.norm(tr-br)), int(np.linalg.norm(tl-br)))

        # Audit fix CORE-12: if two adjacent detected marker centres
        # coincide (degenerate ArUco refinement on a tiny / heavily
        # compressed image) max_width or max_height can be 0, which would
        # raise ``cv2.error: dsize.width and dsize.height are both 0`` in
        # warpPerspective. Bail out with a clear error so the worker logs
        # which sheet failed instead of crashing on an OpenCV assertion.
        if max_width <= 0 or max_height <= 0:
            raise ValueError(
                "four_point_transform: degenerate quadrilateral "
                f"(max_width={max_width}, max_height={max_height})"
            )

        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array(
            [
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1],
            ],
            dtype="float32",
        )

        transform_matrix = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, transform_matrix, (max_width, max_height))

        # return the warped image
        return warped

    @staticmethod
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")

        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # return the ordered coordinates
        return rect
