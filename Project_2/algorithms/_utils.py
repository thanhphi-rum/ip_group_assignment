"""Các hàm dùng chung cho tiền xử lý ảnh trong Project 2."""

from __future__ import annotations

import cv2
import numpy as np


def validate_image(image: np.ndarray) -> None:
    """Kiểm tra ảnh đầu vào trước khi xử lý."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Ảnh đầu vào không hợp lệ hoặc đang rỗng.")
    if image.ndim not in (2, 3):
        raise ValueError("Ảnh phải có 2 hoặc 3 chiều.")


def to_gray(image: np.ndarray) -> np.ndarray:
    """Chuyển ảnh 1/3/4 kênh sang ảnh xám."""
    validate_image(image)
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[:, :, 0]
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError("Ảnh màu phải có 1, 3 hoặc 4 kênh.")


def to_bgr(image: np.ndarray) -> np.ndarray:
    """Đảm bảo ảnh kết quả có 3 kênh màu BGR để vẽ chú thích."""
    validate_image(image)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.shape[2] == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 3:
        return image.copy()
    raise ValueError("Không hỗ trợ số kênh màu của ảnh.")


def odd_kernel_size(value: int, maximum: int = 99) -> int:
    """Đưa kích thước kernel về số lẻ hợp lệ của OpenCV."""
    size = max(1, min(int(value), maximum))
    if size % 2 == 0:
        size += 1 if size < maximum else -1
    return max(1, size)


def prepare_binary_mask(
    image: np.ndarray,
    threshold: int = 100,
    blur_k: int = 5,
    morph_k: int = 3,
    threshold_method: str = "manual",
    polarity: str = "dark",
    morph_operation: str = "close",
) -> tuple[np.ndarray, int]:
    """Tạo mặt nạ nhị phân từ ảnh.

    Args:
        threshold: ngưỡng 0..255 khi dùng phương pháp manual.
        blur_k: kích thước kernel Gaussian (tự chuẩn hóa thành số lẻ).
        morph_k: kích thước kernel morphology; 1 tương đương không lọc.
        threshold_method: ``manual`` hoặc ``otsu``.
        polarity: ``dark`` nếu lỗi tối hơn nền, ``bright`` nếu lỗi sáng hơn nền.
        morph_operation: ``close``, ``open``, ``close_open`` hoặc ``none``.

    Returns:
        (mask, used_threshold). Mặt nạ có giá trị 0 hoặc 255.
    """
    gray = to_gray(image)

    blur_size = odd_kernel_size(blur_k)
    if blur_size > 1:
        gray_for_threshold = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    else:
        gray_for_threshold = gray

    threshold_type = cv2.THRESH_BINARY_INV if polarity == "dark" else cv2.THRESH_BINARY
    method = str(threshold_method).lower()
    if method == "otsu":
        used_threshold, mask = cv2.threshold(
            gray_for_threshold, 0, 255, threshold_type | cv2.THRESH_OTSU
        )
    else:
        used_threshold = max(0, min(255, int(threshold)))
        _, mask = cv2.threshold(
            gray_for_threshold, used_threshold, 255, threshold_type
        )

    kernel_size = odd_kernel_size(morph_k)
    operation = str(morph_operation).lower()
    if kernel_size > 1 and operation != "none":
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_size, kernel_size)
        )
        if operation == "close":
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        elif operation == "open":
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        elif operation == "close_open":
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        else:
            raise ValueError(
                "morph_operation phải là close, open, close_open hoặc none."
            )

    return mask, int(round(float(used_threshold)))
