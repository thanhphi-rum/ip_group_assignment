from algorithms.detect_edges import edges
from algorithms.contour import contour_detect
from algorithms.bounding_box import detect_bounding_boxes

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QSpinBox, QFileDialog,
    QGroupBox, QFormLayout, QCheckBox, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chủ đề 5: Project 2 - Phát hiện lỗi bề mặt")
        self.resize(1180, 680)
        self.setup_stylesheet()

        self.current_image = None
        self.last_result_image = None
        self.video_cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_video_frame)
        self.is_video_playing = False

        self.init_ui()

    def setup_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e2e; }
            QWidget {
                color: #cdd6f4;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 13px;
            }
            QGroupBox {
                border: 1px solid #45475a;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
                font-weight: bold;
                color: #89b4fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #313244;
                border: 1px solid #45475a;
                border-radius: 6px;
                padding: 8px 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #45475a;
                border-color: #89b4fa;
            }
            QPushButton#primaryBtn {
                background-color: #89b4fa;
                color: #11111b;
                border: none;
            }
            QPushButton#primaryBtn:hover { background-color: #b4befe; }
            QComboBox, QSpinBox {
                background-color: #313244;
                border: 1px solid #45475a;
                border-radius: 5px;
                padding: 5px 8px;
                color: #cdd6f4;
            }
            QLabel#imageBox {
                background-color: #181825;
                border: 2px dashed #45475a;
                border-radius: 10px;
                color: #6c7086;
                font-size: 15px;
            }
            QLabel#infoBadge {
                background-color: #313244;
                border-radius: 6px;
                padding: 8px;
                color: #a6e3a1;
                font-weight: bold;
                font-size: 13px;
            }
        """)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        # CỘT TRÁI
        sidebar = QWidget()
        sidebar.setFixedWidth(310)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(12)

        input_group = QGroupBox("1. Nguồn dữ liệu")
        input_layout = QVBoxLayout()
        self.btn_load_img = QPushButton("Tải Ảnh lên")
        self.btn_load_vid = QPushButton("Tải Video lên")
        self.btn_pause_vid = QPushButton("Tạm dừng Video")
        self.btn_pause_vid.setVisible(False)

        self.btn_load_img.clicked.connect(self.load_image)
        self.btn_load_vid.clicked.connect(self.load_video)
        self.btn_pause_vid.clicked.connect(self.toggle_video)

        input_layout.addWidget(self.btn_load_img)
        input_layout.addWidget(self.btn_load_vid)
        input_layout.addWidget(self.btn_pause_vid)
        input_group.setLayout(input_layout)

        algo_group = QGroupBox("2. Chức năng xử lý")
        algo_layout = QVBoxLayout()
        self.combo_algo = QComboBox()
        self.combo_algo.addItems([
            "Khoanh vùng lỗi (Bounding Box)",
            "Phát hiện đường viền (Contour)",
            "Phát hiện biên Canny (Detect Edges)"
        ])
        self.combo_algo.currentIndexChanged.connect(self.on_algo_changed)
        algo_layout.addWidget(self.combo_algo)
        algo_group.setLayout(algo_layout)

        self.param_group = QGroupBox("3. Tham số thuật toán")
        self.param_layout = QFormLayout()
        self.param_layout.setLabelAlignment(Qt.AlignLeft)

        self.lbl_thresh = QLabel("Threshold (0-255):")
        self.spin_thresh = QSpinBox()
        self.spin_thresh.setRange(0, 255)
        self.spin_thresh.setValue(100)
        self.spin_thresh.valueChanged.connect(self.trigger_update)

        self.lbl_min_area = QLabel("Diện tích tối thiểu (Min Area):")
        self.spin_min_area = QSpinBox()
        self.spin_min_area.setRange(1, 100000)
        self.spin_min_area.setSingleStep(10)
        self.spin_min_area.setValue(30)
        self.spin_min_area.valueChanged.connect(self.trigger_update)

        self.chk_show_binary = QCheckBox("Xem ảnh nhị phân (Binary Mask)")
        self.chk_show_binary.stateChanged.connect(self.trigger_update)

        self.lbl_blur_k = QLabel("Blur Kernel (Số lẻ):")
        self.spin_blur_k = QSpinBox()
        self.spin_blur_k.setRange(1, 51)
        self.spin_blur_k.setSingleStep(2)
        self.spin_blur_k.setValue(5)
        self.spin_blur_k.valueChanged.connect(self.on_odd_spin_changed)

        self.lbl_morph_k = QLabel("Morphology Kernel:")
        self.spin_morph_k = QSpinBox()
        self.spin_morph_k.setRange(1, 31)
        self.spin_morph_k.setValue(3)
        self.spin_morph_k.valueChanged.connect(self.trigger_update)

        self.lbl_low_thresh = QLabel("Low Threshold:")
        self.spin_low_thresh = QSpinBox()
        self.spin_low_thresh.setRange(0, 254)
        self.spin_low_thresh.setValue(80)
        self.spin_low_thresh.valueChanged.connect(self.trigger_update)

        self.lbl_high_thresh = QLabel("High Threshold:")
        self.spin_high_thresh = QSpinBox()
        self.spin_high_thresh.setRange(1, 255)
        self.spin_high_thresh.setValue(160)
        self.spin_high_thresh.valueChanged.connect(self.trigger_update)

        self.param_layout.addRow(self.lbl_blur_k, self.spin_blur_k)
        self.param_layout.addRow(self.lbl_thresh, self.spin_thresh)
        self.param_layout.addRow(self.lbl_morph_k, self.spin_morph_k)
        self.param_layout.addRow(self.lbl_min_area, self.spin_min_area)
        self.param_layout.addRow(self.chk_show_binary)
        self.param_layout.addRow(self.lbl_low_thresh, self.spin_low_thresh)
        self.param_layout.addRow(self.lbl_high_thresh, self.spin_high_thresh)
        self.param_group.setLayout(self.param_layout)

        info_group = QGroupBox("4. Thống kê kết quả")
        info_layout = QVBoxLayout()
        self.lbl_info = QLabel("Chưa có dữ liệu xử lý")
        self.lbl_info.setObjectName("infoBadge")
        self.lbl_info.setAlignment(Qt.AlignCenter)
        self.lbl_info.setWordWrap(True)

        self.btn_refresh = QPushButton("Cập nhật kết quả")
        self.btn_refresh.setObjectName("primaryBtn")
        self.btn_refresh.clicked.connect(self.update_image_result)

        info_layout.addWidget(self.lbl_info)
        info_layout.addWidget(self.btn_refresh)
        info_group.setLayout(info_layout)

        sidebar_layout.addWidget(input_group)
        sidebar_layout.addWidget(algo_group)
        sidebar_layout.addWidget(self.param_group)
        sidebar_layout.addWidget(info_group)
        sidebar_layout.addStretch()

        # CỘT PHẢI (KHÓA KÍCH THƯỚC CHỐNG PHÌNH TO)
        display_layout = QHBoxLayout()
        display_layout.setSpacing(14)

        orig_container = QVBoxLayout()
        lbl_orig_title = QLabel("ẢNH / VIDEO GỐC")
        lbl_orig_title.setAlignment(Qt.AlignCenter)
        lbl_orig_title.setFixedHeight(24)
        lbl_orig_title.setStyleSheet("font-weight: bold; color: #a6adc8;")

        self.lbl_orig_view = QLabel("Chưa tải dữ liệu")
        self.lbl_orig_view.setObjectName("imageBox")
        self.lbl_orig_view.setAlignment(Qt.AlignCenter)
        self.lbl_orig_view.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.lbl_orig_view.setMinimumSize(320, 320)

        orig_container.addWidget(lbl_orig_title)
        orig_container.addWidget(self.lbl_orig_view, stretch=1)

        res_container = QVBoxLayout()
        lbl_res_title = QLabel("KẾT QUẢ XỬ LÝ")
        lbl_res_title.setAlignment(Qt.AlignCenter)
        lbl_res_title.setFixedHeight(24)
        lbl_res_title.setStyleSheet("font-weight: bold; color: #89b4fa;")

        self.lbl_res_view = QLabel("Chưa có kết quả")
        self.lbl_res_view.setObjectName("imageBox")
        self.lbl_res_view.setAlignment(Qt.AlignCenter)
        self.lbl_res_view.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.lbl_res_view.setMinimumSize(320, 320)

        res_container.addWidget(lbl_res_title)
        res_container.addWidget(self.lbl_res_view, stretch=1)

        display_layout.addLayout(orig_container, stretch=1)
        display_layout.addLayout(res_container, stretch=1)

        main_layout.addWidget(sidebar)
        main_layout.addLayout(display_layout, stretch=1)

        self.on_algo_changed(0)

    def on_algo_changed(self, index):
        is_bbox = (index == 0)
        is_contour = (index == 1)
        is_edges = (index == 2)

        self.spin_thresh.blockSignals(True)
        self.spin_min_area.blockSignals(True)
        if is_bbox:
            self.spin_thresh.setValue(100)
            self.spin_min_area.setValue(30)
        elif is_contour:
            self.spin_thresh.setValue(127)
            self.spin_min_area.setValue(100)
        self.spin_thresh.blockSignals(False)
        self.spin_min_area.blockSignals(False)

        self.lbl_thresh.setVisible(is_bbox or is_contour)
        self.spin_thresh.setVisible(is_bbox or is_contour)
        self.lbl_min_area.setVisible(is_bbox or is_contour)
        self.spin_min_area.setVisible(is_bbox or is_contour)

        self.chk_show_binary.setVisible(is_bbox)

        self.lbl_blur_k.setVisible(is_contour)
        self.spin_blur_k.setVisible(is_contour)
        self.lbl_morph_k.setVisible(is_contour)
        self.spin_morph_k.setVisible(is_contour)

        self.lbl_low_thresh.setVisible(is_edges)
        self.spin_low_thresh.setVisible(is_edges)
        self.lbl_high_thresh.setVisible(is_edges)
        self.spin_high_thresh.setVisible(is_edges)

        self.trigger_update()

    def on_odd_spin_changed(self, val):
        if val % 2 == 0:
            self.spin_blur_k.setValue(val + 1)
            return
        self.trigger_update()

    def trigger_update(self):
        if self.current_image is not None and not self.is_video_playing:
            self.update_image_result()

    def stop_video_if_running(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None
        self.is_video_playing = False
        self.btn_pause_vid.setVisible(False)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Chọn Ảnh", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if not file_path:
            return

        self.stop_video_if_running()
        img_array = np.fromfile(file_path, dtype=np.uint8)
        self.current_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if self.current_image is not None:
            self.display_cv_image(self.current_image, self.lbl_orig_view)
            self.update_image_result()

    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Chọn Video", "", "Video Files (*.mp4 *.avi *.mkv *.mov)"
        )
        if not file_path:
            return

        self.stop_video_if_running()
        self.current_image = None
        self.video_cap = cv2.VideoCapture(file_path)

        if self.video_cap.isOpened():
            self.is_video_playing = True
            self.btn_pause_vid.setVisible(True)
            self.btn_pause_vid.setText("Tạm dừng Video")
            self.timer.start(30)

    def toggle_video(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_pause_vid.setText("Tiếp tục Video")
        else:
            self.timer.start(30)
            self.btn_pause_vid.setText("Tạm dừng Video")

    def apply_selected_algorithm(self, frame_bgr):
        algo_idx = self.combo_algo.currentIndex()

        if algo_idx == 0:
            t_val = self.spin_thresh.value()
            min_a = self.spin_min_area.value()
            result, info = detect_bounding_boxes(frame_bgr, threshold=t_val, min_area=min_a)
            self.lbl_info.setText(f"Phát hiện: {info['boxes']} vùng nghi lỗi")
            if self.chk_show_binary.isChecked():
                return info["binary"]
            return result

        elif algo_idx == 1:
            b_k = self.spin_blur_k.value()
            t_val = self.spin_thresh.value()
            m_k = self.spin_morph_k.value()
            min_a = self.spin_min_area.value()
            result, count = contour_detect(
                frame_bgr, blur_k=b_k, thresh=t_val, morph_k=m_k, min_erea=min_a
            )
            self.lbl_info.setText(f"Số đường viền (Contours): {count}")
            return result

        elif algo_idx == 2:
            low_t = self.spin_low_thresh.value()
            high_t = self.spin_high_thresh.value()
            result, edge_px = edges(frame_bgr, low_threshold=low_t, high_threshold=high_t)
            self.lbl_info.setText(f"Số điểm ảnh biên (Edge Pixels): {edge_px:,}")
            return result

    def update_image_result(self):
        if self.current_image is not None:
            res = self.apply_selected_algorithm(self.current_image)
            self.last_result_image = res
            self.display_cv_image(res, self.lbl_res_view)

    def process_video_frame(self):
        if self.video_cap is not None and self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            if not ret:
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return

            self.current_image = frame
            self.display_cv_image(frame, self.lbl_orig_view)
            res = self.apply_selected_algorithm(frame)
            self.last_result_image = res
            self.display_cv_image(res, self.lbl_res_view)

    def display_cv_image(self, img, label_widget):
        if img is None:
            return

        if img.ndim == 2:
            h, w = img.shape
            q_img = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        else:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_img.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)

        target_size = label_widget.contentsRect().size()
        if target_size.width() > 10 and target_size.height() > 10:
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(
                target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            label_widget.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_image is not None and not self.timer.isActive():
            self.display_cv_image(self.current_image, self.lbl_orig_view)
            if self.last_result_image is not None:
                self.display_cv_image(self.last_result_image, self.lbl_res_view)