"""Các thuật toán xử lý ảnh của Project 2."""

from .detect_edges import edges
from .contour import contour_detect
from .bounding_box import detect_bounding_boxes

__all__ = ["edges", "contour_detect", "detect_bounding_boxes"]
