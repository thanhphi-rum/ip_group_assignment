import numpy as np

def otsu_threshold(img_gray):
    # -----------------------------------------------------
    # BƯỚC 1: TẠO HISTOGRAM
    # -----------------------------------------------------
    # Histogram gồm 256 phần tử
    histogram = np.zeros(256, dtype=np.int64)

    for pixel in img_gray.ravel():
        histogram[pixel] += 1

    # Tổng số pixel của ảnh
    total_pixels = img_gray.size

    # -----------------------------------------------------
    # BƯỚC 2: TÍNH TỔNG MỨC XÁM
    # -----------------------------------------------------
    total_sum = 0

    for i in range(256):
        total_sum += i * histogram[i]

    # -----------------------------------------------------
    # BƯỚC 3: KHỞI TẠO GIÁ TRỊ
    # -----------------------------------------------------
    weight_background = 0
    sumgray_background = 0
    max_variance = 0
    best_threshold = 0

    # -----------------------------------------------------
    # BƯỚC 4: THỬ TỪNG NGƯỠNG
    # -----------------------------------------------------
    for t in range(256):

        weight_background += histogram[t]

        if weight_background == 0:
            continue

        weight_foreground = total_pixels - weight_background

        if weight_foreground == 0:
            break

        sumgray_background += t * histogram[t]

        # -------------------------------------------------
        # BƯỚC 5: TÍNH TRUNG BÌNH MỖI NHÓM
        # -------------------------------------------------
        mean_background = (
            sumgray_background / weight_background
        )

        mean_foreground = (
            (total_sum - sumgray_background)
            / weight_foreground
        )

        # -------------------------------------------------
        # BƯỚC 6: TÍNH PHƯƠNG SAI GIỮA HAI NHÓM
        # -------------------------------------------------
        variance = (
            weight_background
            * weight_foreground
            * (mean_background - mean_foreground) ** 2
        )

        # -------------------------------------------------
        # BƯỚC 7: TÌM NGƯỠNG TỐT NHẤT
        # -------------------------------------------------
        if variance > max_variance:
            max_variance = variance
            best_threshold = t

    # Trả về ngưỡng Otsu tìm được
    return best_threshold