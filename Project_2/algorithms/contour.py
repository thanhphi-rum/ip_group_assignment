import cv2
from ._utils import prepare_binary_mask, to_bgr, validate_image


def contour_detect(img, blur_k=5, thresh=127, morph_k=3, min_area=100, threshold_method="manual", polarity="dark", morph_operation="close", min_length=0, return_contours=False, min_erea=None):
    validate_image(img)
    if min_erea is not None:
        min_area = min_erea

    mask, _ = prepare_binary_mask(img, threshold=thresh, blur_k=blur_k, morph_k=morph_k, threshold_method=threshold_method, polarity=polarity, morph_operation=morph_operation)

    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_area = img.shape[0] * img.shape[1]
    min_area = max(0.0, float(min_area))
    min_length = max(0.0, float(min_length))

    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > image_area * 0.90:
            continue
        is_big = area >= min_area
        is_long = min_length > 0 and cv2.arcLength(contour, True) >= min_length
        if is_big or is_long:
            valid_contours.append(contour)

    result = to_bgr(img)
    if valid_contours:
        cv2.drawContours(result, valid_contours, -1, (0, 0, 255), 2)

    if return_contours:
        return result, len(valid_contours), valid_contours
    return result, len(valid_contours)