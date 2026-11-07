import cv2

def edges(image, low_threshold=80, high_threshold=160):
    """Phát hiện biên bằng thuật toán Canny."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    low = int(low_threshold)
    high = int(high_threshold)
    if low >= high:
        high = min(255, low + 1)
    edges = cv2.Canny(gray, low, high)
    result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edge_pixels = cv2.countNonZero(edges)
    return result, edge_pixels
