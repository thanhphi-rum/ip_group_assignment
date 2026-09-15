import numpy as np

def adaptive_threshold(img_gray, block_size=11, C=2):
    # Kiểm tra block_size c là số chẵn, thì tăng lên 1 để đảm bảo là số lẻ
    if block_size % 2 == 0:
        block_size += 1 # block_size phải là số lẻ để đảm bảo có một pixel trung tâm

    h, w = img_gray.shape # Lấy height và width của ảnh xám

    # Padding bằng reflect để tránh lỗi khi tính toán ngưỡng cho các pixel ở biên
    pad_size = block_size // 2 
    padded_img = np.pad(img_gray, pad_size, mode='reflect').astype(np.float32)

    result = np.zeros(img_gray.shape, dtype=np.uint8)
    # Tính toán integral image để tăng tốc độ tính toán mean
    integral_img = np.cumsum(np.cumsum(padded_img, axis=0), axis=1)
    integral_img = np.pad(integral_img, ((1,0), (1, 0)), mode="constant")
    # Duyệt từng pixel
    for i in range(h):
        for j in range(w):
            y1, y2 = i, i + block_size
            x1, x2 = j,j + block_size
            total = integral_img[y2, x2] - integral_img[y1, x2] - integral_img[y2, x1] + integral_img[y1, x1]
            mean = total / (block_size * block_size) # Mean vùng lân cận
            threshold = mean - C
            result[i, j] = 255 if padded_img[i + pad_size, j + pad_size] > threshold else 0 # Nếu pixel lớn hơn ngưỡng thì gán giá trị 255, ngược lại gán giá trị 0
    return result

def adaptive_gaussian(img_gray, block_size=11, C=2, sigma=None):
    # Đảm bảo block_size là số lẻ
    if block_size % 2 == 0:
        block_size += 1

    # Lấy kích thước ảnh
    h, w = img_gray.shape

    # Tính kích thước padding
    pad_size = block_size // 2

    # Padding ảnh bằng reflect
    padded_img = np.pad(img_gray, pad_size, mode='reflect').astype(np.float32)

    # Nếu không truyền sigma thì tự động tính
    if sigma is None:
        sigma = 0.3 * ((block_size - 1) * 0.5 - 1) + 0.8

    # Tạo Gaussian kernel
    ax = np.arange(-pad_size, pad_size + 1)
    xx, yy = np.meshgrid(ax, ax)
    gaussian_kernel = np.exp(
        -(xx**2 + yy**2) / (2 * sigma**2)
    )

    # Chuẩn hóa kernel để tổng trọng số = 1
    gaussian_kernel /= gaussian_kernel.sum()

    # Ma trận kết quả
    result = np.zeros(img_gray.shape, dtype=np.uint8)
    
    # Duyệt từng pixel
    for i in range(h):
        for j in range(w):
            # Lấy cửa sổ lân cận
            window = padded_img[i:i + block_size, j:j + block_size]
            # Tính Gaussian Weighted Mean
            weighted_mean = np.sum(window * gaussian_kernel)
            # Tính threshold
            threshold = weighted_mean - C
            # Pixel hiện tại trong ảnh padded
            pixel = padded_img[i + pad_size, j + pad_size]
            # Phân loại pixel
            result[i, j] = (255 if pixel > threshold else 0)
    return result