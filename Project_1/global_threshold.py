import numpy as np

def global_threshold(img_gray, threshold=127):

    # Tạo ma trận kết quả có kích thước giống ảnh xám
    result = np.zeros(img_gray.shape,dtype=np.uint8)
    # Duyệt qua toàn bộ pixel của ảnh
    # Nếu pixel lớn hơn ngưỡng thì gán màu trắng
    result[img_gray > threshold] = 255

    # Trả về ảnh nhị phân
    return result