import cv2
import numpy as np

def contour_detect(img, blur_k=5, thresh=127, morph_k=3, min_erea=100):
    # B1. Chuyển sang ảnh xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # B2. Làm mờ để giảm nhiễu
    blur = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    # B3. Phân ngưỡng
    _, mask = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY_INV)
    # B4. Morphology: xóa chấm nhỏ, lấp lỗ hỏng
    kernel = np.ones((morph_k, morph_k), np.uint8)
    # B5. Tìm contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # B6. Giữ contour đủ lớn
    contours = [c for c in contours if cv2.contourArea(c) >= min_erea]
    # B7. Vẽ lên ảnh gốc
    result = img.copy()
    cv2.drawContours(result, contours, -1, (0, 0, 255), 2)
    return result, len(contours)