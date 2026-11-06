"""Phát hiện đường biên bằng thuật toán Canny."""

import cv2
import numpy as np

from ._utils import odd_kernel_size, to_bgr, to_gray, validate_image


def edges(image, low_threshold=50, high_threshold=150, blur_k=5):
    """Phát hiện biên Canny và tô đường biên màu đỏ lên ảnh gốc.

    Returns:
        result: ảnh BGR đã đánh dấu đường biên.
        edge_pixels: số điểm ảnh biên được phát hiện.
    """
    validate_image(image)
    gray = to_gray(image)

    blur_size = odd_kernel_size(blur_k)
    if blur_size > 1:
        gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

    low = max(0, min(255, int(low_threshold)))
    high = max(0, min(255, int(high_threshold)))
    if low > high:
        low, high = high, low
    if low == high:
        high = min(255, low + 1)
        if low == high:  # Trường hợp cả hai ngưỡng đều bằng 255.
            low = 254

    edge_map = cv2.Canny(gray, low, high)
    result = to_bgr(image)
    result[edge_map > 0] = (0, 0, 255)  # Đánh dấu biên màu đỏ.
    edge_pixels = int(cv2.countNonZero(edge_map))
    return result, edge_pixels
