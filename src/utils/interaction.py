from dataclasses import dataclass

import cv2

from src.logger import logger
from src.utils.image import ImageUtils

# Headless-safe monitor probe. screeninfo raises in environments without a
# display server (Docker, CI, SSH without X forwarding), so swallow any
# failure and fall back to a sane virtual desktop size. Display features
# (cv2.imshow / cv2.moveWindow) are no-ops in headless mode anyway.
try:
    from screeninfo import get_monitors

    _monitors = get_monitors()
    monitor_window = _monitors[0] if _monitors else None
except Exception as _exc:  # pragma: no cover - environment-dependent
    logger.warning(
        f"screeninfo unavailable ({_exc!r}); falling back to virtual 1920x1080 desktop"
    )
    monitor_window = None

if monitor_window is not None:
    _DEFAULT_WINDOW_WIDTH = monitor_window.width
    _DEFAULT_WINDOW_HEIGHT = monitor_window.height
else:
    _DEFAULT_WINDOW_WIDTH = 1920
    _DEFAULT_WINDOW_HEIGHT = 1080


@dataclass
class ImageMetrics:
    # TODO: Move TEXT_SIZE, etc here and find a better class name
    window_width, window_height = _DEFAULT_WINDOW_WIDTH, _DEFAULT_WINDOW_HEIGHT
    # for positioning image windows
    window_x, window_y = 0, 0
    reset_pos = [0, 0]


class InteractionUtils:
    """Perform primary functions such as displaying images and reading responses"""

    image_metrics = ImageMetrics()

    @staticmethod
    def show(name, origin, pause=1, resize=False, reset_pos=None, config=None):
        image_metrics = InteractionUtils.image_metrics
        if origin is None:
            logger.info(f"'{name}' - NoneType image to show!")
            if pause:
                cv2.destroyAllWindows()
            return
        if resize:
            if not config:
                raise Exception("config not provided for resizing the image to show")
            img = ImageUtils.resize_util(origin, config.dimensions.display_width)
        else:
            img = origin

        if not is_window_available(name):
            cv2.namedWindow(name)

        cv2.imshow(name, img)

        if reset_pos:
            image_metrics.window_x = reset_pos[0]
            image_metrics.window_y = reset_pos[1]

        cv2.moveWindow(
            name,
            image_metrics.window_x,
            image_metrics.window_y,
        )

        h, w = img.shape[:2]

        # Set next window position
        margin = 25
        w += margin
        h += margin

        w, h = w // 2, h // 2
        if image_metrics.window_x + w > image_metrics.window_width:
            image_metrics.window_x = 0
            if image_metrics.window_y + h > image_metrics.window_height:
                image_metrics.window_y = 0
            else:
                image_metrics.window_y += h
        else:
            image_metrics.window_x += w

        if pause:
            logger.info(
                f"Showing '{name}'\n\t Press Q on image to continue. Press Ctrl + C in terminal to exit"
            )

            wait_q()
            InteractionUtils.image_metrics.window_x = 0
            InteractionUtils.image_metrics.window_y = 0


@dataclass
class Stats:
    # TODO Fill these for stats
    # Move qbox_vals here?
    # badThresholds = []
    # veryBadPoints = []
    files_moved = 0
    files_not_moved = 0


def wait_q():
    esc_key = 27
    while cv2.waitKey(1) & 0xFF not in [ord("q"), esc_key]:
        pass
    cv2.destroyAllWindows()


def is_window_available(name: str) -> bool:
    """Checks if a window is available"""
    try:
        cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE)
        return True
    except Exception as e:
        print(e)
        return False
