from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage, QFont
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox, QTabWidget,
    QVBoxLayout, QHBoxLayout, QFormLayout, QFileDialog, QMessageBox,
    QScrollArea, QSizePolicy
)

import numpy as np

from global_threshold import global_threshold
from otsu_threshold import otsu_threshold
from adaptive_threshold import adaptive_threshold, adaptive_gaussian


class ImageBox(QGroupBox):
    def __init__(self, title):
        super().__init__(title)

        layout = QVBoxLayout()

        self.label = QLabel("Chưa có ảnh")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumSize(420, 330)
        self.label.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )

        self.label.setStyleSheet("""
            QLabel {
                border: 1px dashed #9aa0a6;
                background: #fafafa;
                color: #777;
                font-size: 16px;
            }
        """)

        layout.addWidget(self.label)
        self.setLayout(layout)

    def show_text(self, text):
        self.label.clear()
        self.label.setText(text)

    def show_pixmap(self, pixmap):
        self.label.setPixmap(
            pixmap.scaled(
                self.label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Ứng dụng xử lý ảnh - Chủ đề 5")
        self.resize(1250, 820)

        self.original_image = None
        self.original_video = None
        self.result = None

        self.setup_style()
        self.setup_ui()

    # ==========================================================
    # STYLE
    # ==========================================================

    def setup_style(self):
        self.setStyleSheet("""
            QMainWindow {
                background: #f4f6f8;
            }

            QGroupBox {
                background: white;
                border: 1px solid #cfd4da;
                border-radius: 8px;
                margin-top: 12px;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }

            QLabel {
                font-size: 14px;
            }

            QPushButton {
                min-height: 38px;
                padding: 5px 15px;
                border-radius: 6px;
                font-size: 14px;
            }

            QPushButton:hover {
                background: #e9ecef;
            }

            QComboBox,
            QSpinBox,
            QDoubleSpinBox {
                min-height: 34px;
                font-size: 14px;
                padding: 0 8px;
                border: 1px solid #cfd4da;
                border-radius: 5px;
                background: white;
            }


            QTabBar::tab {
                padding: 12px 25px;
                font-size: 14px;
            }
        """)

    # ==========================================================
    # GIAO DIỆN CHÍNH
    # ==========================================================

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(15, 12, 15, 12)

        title = QLabel("ỨNG DỤNG XỬ LÝ ẢNH - CHỦ ĐỀ 5")
        title.setAlignment(Qt.AlignCenter)

        font = QFont()
        font.setPointSize(22)
        font.setBold(True)
        title.setFont(font)

        root.addWidget(title)

        tabs = QTabWidget()
        tabs.addTab(self.create_threshold_tab(), "Ngưỡng hóa ảnh")
        root.addWidget(tabs)

    # ==========================================================
    # THANH NÚT CHUNG
    # ==========================================================

    def create_file_buttons(self, open_image_slot, open_video_slot,
                            save_slot, process_slot):
        box = QGroupBox("Tệp và xử lý")
        layout = QHBoxLayout()

        btn_image = QPushButton("Mở ảnh")
        btn_video = QPushButton("Mở video")
        btn_process = QPushButton("Chạy xử lý")
        btn_save = QPushButton("Lưu kết quả")

        btn_image.clicked.connect(open_image_slot)
        btn_video.clicked.connect(open_video_slot)
        btn_process.clicked.connect(process_slot)
        btn_save.clicked.connect(save_slot)

        layout.addWidget(btn_image)
        layout.addWidget(btn_video)
        layout.addStretch()
        layout.addWidget(btn_process)
        layout.addWidget(btn_save)

        box.setLayout(layout)
        return box

    # ==========================================================
    # TAB 1 - NGƯỠNG HÓA ẢNH
    # ==========================================================

    def create_threshold_tab(self):
        page = QWidget()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(10)

        file_box = self.create_file_buttons(
            self.open_threshold_image,
            self.open_threshold_video,
            self.save_threshold,
            self.run_threshold
        )
        layout.addWidget(file_box)

        images = QHBoxLayout()

        self.threshold_original = ImageBox("ẢNH GỐC")
        self.threshold_result = ImageBox("ẢNH SAU XỬ LÝ")

        images.addWidget(self.threshold_original, 1)
        images.addWidget(self.threshold_result, 1)
        layout.addLayout(images)

        # -------------------------------
        # THUẬT TOÁN + THAM SỐ
        # -------------------------------

        setting_row = QHBoxLayout()

        algorithm_box = QGroupBox("Phương pháp")
        algorithm_layout = QVBoxLayout()

        self.threshold_method = QComboBox()
        self.threshold_method.addItems([
            "Global Threshold",
            "Otsu",
            "Adaptive Threshold"
        ])
        self.threshold_method.currentIndexChanged.connect(
            self.update_threshold_parameters
        )

        algorithm_layout.addWidget(QLabel("Chọn phương pháp:"))
        algorithm_layout.addWidget(self.threshold_method)
        algorithm_layout.addStretch()
        algorithm_box.setLayout(algorithm_layout)

        self.threshold_parameter_box = QGroupBox("Tham số")
        parameter_layout = QFormLayout()

        self.global_threshold_label = QLabel("Threshold:")
        self.global_threshold = QSpinBox()
        self.global_threshold.setRange(0, 255)
        self.global_threshold.setValue(128)

        self.adaptive_method_label = QLabel("Phương pháp:")
        self.adaptive_method = QComboBox()
        self.adaptive_method.addItems(["Mean", "Gaussian"])

        self.block_size_label = QLabel("Block Size:")
        self.block_size = QSpinBox()
        self.block_size.setRange(3, 99)
        self.block_size.setSingleStep(2)
        self.block_size.setValue(11)

        self.c_value_label = QLabel("C:")
        self.c_value = QDoubleSpinBox()
        self.c_value.setRange(-100, 100)
        self.c_value.setDecimals(2)
        self.c_value.setValue(2)

        parameter_layout.addRow(
            self.global_threshold_label,
            self.global_threshold
        )
        parameter_layout.addRow(
            self.adaptive_method_label,
            self.adaptive_method
        )
        parameter_layout.addRow(
            self.block_size_label,
            self.block_size
        )
        parameter_layout.addRow(
            self.c_value_label,
            self.c_value
        )

        self.threshold_parameter_box.setLayout(parameter_layout)

        setting_row.addWidget(algorithm_box, 1)
        setting_row.addWidget(self.threshold_parameter_box, 2)
        layout.addLayout(setting_row)

        layout.addStretch()

        scroll.setWidget(content)

        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.addWidget(scroll)

        self.update_threshold_parameters()
        return page

    # ==========================================================
    # ẨN / HIỆN THAM SỐ NGƯỠNG HÓA
    # ==========================================================

    def update_threshold_parameters(self):
        method = self.threshold_method.currentText()

        is_global = method == "Global Threshold"
        is_otsu = method == "Otsu"
        is_adaptive = method == "Adaptive Threshold"

        self.global_threshold_label.setVisible(is_global)
        self.global_threshold.setVisible(is_global)

        self.adaptive_method_label.setVisible(is_adaptive)
        self.adaptive_method.setVisible(is_adaptive)

        self.block_size_label.setVisible(is_adaptive)
        self.block_size.setVisible(is_adaptive)

        self.c_value_label.setVisible(is_adaptive)
        self.c_value.setVisible(is_adaptive)

    # ==========================================================
    # MỞ ẢNH
    # ==========================================================

    def open_threshold_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn ảnh",
            "",
            "Ảnh (*.jpg *.jpeg *.png *.bmp)"
        )

        if not path:
            return

        pixmap = QPixmap(path)
        if pixmap.isNull():
            QMessageBox.warning(
                self,
                "Lỗi",
                "Không thể đọc ảnh đã chọn."
            )
            return

        self.original_image = path
        self.original_video = None
        self.result = None

        self.threshold_original.show_pixmap(pixmap)
        self.threshold_result.show_text("Chưa có ảnh")

    # ==========================================================
    # MỞ VIDEO
    # ==========================================================

    def open_threshold_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn video",
            "",
            "Video (*.mp4 *.avi *.mov *.mkv)"
        )

        if not path:
            return

        self.original_video = path
        self.original_image = None
        self.result = None

        self.threshold_original.show_text(
            f"VIDEO ĐÃ CHỌN\n\n{path}"
        )
        self.threshold_result.show_text("Chưa xử lý video")

    # ==========================================================
    # CHẠY XỬ LÝ
    # ==========================================================

    def run_threshold(self):
        if not self.original_image:
            QMessageBox.warning(
                self,
                "Thông báo",
                "Vui lòng mở ảnh trước khi chạy xử lý."
            )
            return

        try:
            image = QImage(self.original_image)

            if image.isNull():
                raise ValueError("Không thể đọc ảnh.")

            image = image.convertToFormat(QImage.Format_Grayscale8)

            width = image.width()
            height = image.height()

            ptr = image.bits()
            img_gray = np.frombuffer(
                ptr,
                dtype=np.uint8,
                count=height * width
            ).reshape((height, width))

            method = self.threshold_method.currentText()
            threshold = None

            if method == "Global Threshold":
                threshold = self.global_threshold.value()
                result = global_threshold(img_gray, threshold)

            elif method == "Otsu":
                threshold = otsu_threshold(img_gray)
                result = global_threshold(img_gray, threshold)

            elif method == "Adaptive Threshold":
                adaptive_method = self.adaptive_method.currentText()
                block_size = self.block_size.value()
                c_value = self.c_value.value()

                if adaptive_method == "Mean":
                    result = adaptive_threshold(
                        img_gray,
                        block_size,
                        c_value
                    )
                else:
                    result = adaptive_gaussian(
                        img_gray,
                        block_size,
                        c_value
                    )

            else:
                raise ValueError("Phương pháp xử lý không hợp lệ.")

            # Đảm bảo kết quả là ma trận uint8 2D
            result = np.asarray(result, dtype=np.uint8)
            if result.shape != (height, width):
                raise ValueError("Kích thước ảnh kết quả không hợp lệ.")

            result_image = QImage(
                result.data,
                width,
                height,
                width,
                QImage.Format_Grayscale8
            ).copy()

            pixmap = QPixmap.fromImage(result_image)
            self.threshold_result.show_pixmap(pixmap)
            self.result = result.copy()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Lỗi xử lý",
                f"Không thể xử lý ảnh:\n{e}"
            )

    # ==========================================================
    # LƯU KẾT QUẢ
    # ==========================================================

    def save_threshold(self):
        if self.result is None:
            QMessageBox.warning(
                self,
                "Thông báo",
                "Chưa có kết quả để lưu. Vui lòng xử lý ảnh trước."
            )
            return

        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Lưu kết quả",
            "",
            "PNG (*.png);;JPG (*.jpg)"
        )

        if not path:
            return

        # Nếu người dùng chưa nhập phần mở rộng, tự thêm theo bộ lọc
        if "." not in path.split("/")[-1]:
            if "JPG" in selected_filter:
                path += ".jpg"
            else:
                path += ".png"

        try:
            saved = QImage(
                self.result.data,
                self.result.shape[1],
                self.result.shape[0],
                self.result.shape[1],
                QImage.Format_Grayscale8
            ).copy().save(path)

            if not saved:
                raise IOError("Không thể ghi file kết quả.")

            QMessageBox.information(
                self,
                "Lưu kết quả",
                f"Đã lưu kết quả tại:\n{path}"
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Lỗi lưu kết quả",
                f"Không thể lưu ảnh:\n{e}"
            )
