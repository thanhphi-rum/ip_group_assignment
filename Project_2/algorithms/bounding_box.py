import cv2


def detect_bounding_boxes(image, threshold=100, min_area=30):
    """Tim cac vung nghi loi va ve khung chu nhat bao quanh (Bounding Box)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, int(threshold), 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = image.copy()
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= float(min_area):
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, w, h, area))
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(result, f"{int(area)}", (x, max(18, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return result, {"boxes": len(boxes), "details": boxes, "binary": binary}
