"""Phát hiện vùng nghi lỗi và vẽ Bounding Box."""

import cv2

from ._utils import prepare_binary_mask, to_bgr, validate_image


def detect_bounding_boxes(
    image,
    threshold=100,
    min_area=30,
    blur_k=5,
    morph_k=3,
    threshold_method="manual",
    polarity="dark",
    morph_operation="close",
):
    """Tìm các vùng nghi lỗi và vẽ khung chữ nhật quanh từng vùng.

    Returns:
        result: ảnh màu có các Bounding Box.
        stats: số box, thông tin từng box, mặt nạ và ngưỡng đã dùng.
    """
    validate_image(image)
    binary, used_threshold = prepare_binary_mask(
        image,
        threshold=threshold,
        blur_k=blur_k,
        morph_k=morph_k,
        threshold_method=threshold_method,
        polarity=polarity,
        morph_operation=morph_operation,
    )

    contours, _ = cv2.findContours(
        binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    image_area = image.shape[0] * image.shape[1]
    min_area = max(0.0, float(min_area))
    result = to_bgr(image)
    boxes = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > image_area * 0.90:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        boxes.append((x, y, width, height, area))
        cv2.rectangle(
            result,
            (x, y),
            (x + width, y + height),
            (0, 0, 255),
            2,
        )
        cv2.putText(
            result,
            f"{area:.0f} px",
            (x, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    return result, {
        "boxes": len(boxes),
        "details": boxes,
        "binary": binary,
        "threshold_used": used_threshold,
        "total_area": sum(box[4] for box in boxes),
    }
