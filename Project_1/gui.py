from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage, QFont
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox, QTabWidget,
    QVBoxLayout, QHBoxLayout, QFormLayout, QFileDialog, QMessageBox,
    QScrollArea, QPlainTextEdit, QSizePolicy
)

import numpy as np

from global_threshold import global_threshold
from otsu_threshold import otsu_threshold
from adaptive_threshold import adaptive_threshold


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

            QPlainTextEdit {
                border: 1px solid #cfd4da;
                border-radius: 5px;
                background: white;
                font-size: 14px;
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

        tabs.addTab(
            self.create_threshold_tab(),
            "Ngưỡng hóa ảnh"
        )

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

        # Scroll để có con trỏ kéo lên/xuống
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(10)

        # -------------------------------
        # FILE
        # -------------------------------

        file_box = self.create_file_buttons(
            self.open_threshold_image,
            self.open_threshold_video,
            self.save_threshold,
            self.run_threshold
        )

        layout.addWidget(file_box)

        # -------------------------------
        # ẢNH
        # -------------------------------

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

        algorithm_layout.addWidget(
            QLabel("Chọn phương pháp:")
        )
        algorithm_layout.addWidget(
            self.threshold_method
        )
        algorithm_layout.addStretch()

        algorithm_box.setLayout(algorithm_layout)

        # Parameters
        self.threshold_parameter_box = QGroupBox(
            "Tham số"
        )

        parameter_layout = QFormLayout()

        self.global_threshold = QSpinBox()
        self.global_threshold.setRange(0, 255)
        self.global_threshold.setValue(128)

        self.block_size = QSpinBox()
        self.block_size.setRange(3, 99)
        self.block_size.setSingleStep(2)
        self.block_size.setValue(11)

        self.c_value = QDoubleSpinBox()
        self.c_value.setRange(-100, 100)
        self.c_value.setValue(2)

        parameter_layout.addRow(
            "Threshold:",
            self.global_threshold
        )

        parameter_layout.addRow(
            "Block Size:",
            self.block_size
        )

        parameter_layout.addRow(
            "C:",
            self.c_value
        )

        self.threshold_parameter_box.setLayout(
            parameter_layout
        )

        setting_row.addWidget(algorithm_box, 1)
        setting_row.addWidget(
            self.threshold_parameter_box, 2
        )

        layout.addLayout(setting_row)

        # -------------------------------
        # ĐÁNH GIÁ
        # -------------------------------

        evaluation_box = QGroupBox(
            "Đánh giá kết quả"
        )

        evaluation_layout = QVBoxLayout()

        self.threshold_comment = QPlainTextEdit()
        self.threshold_comment.setPlaceholderText(
            "Nhập nhận xét về kết quả xử lý..."
        )
        self.threshold_comment.setMinimumHeight(100)

        evaluation_layout.addWidget(
            QLabel("Nhận xét:")
        )
        evaluation_layout.addWidget(
            self.threshold_comment
        )

        evaluation_box.setLayout(
            evaluation_layout
        )

        layout.addWidget(evaluation_box)

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

        if method == "Global Threshold":
            self.global_threshold.setVisible(True)
            self.block_size.setVisible(False)
            self.c_value.setVisible(False)

            self.threshold_parameter_box.layout().labelForField(
                self.global_threshold
            ).setVisible(True)

        elif method == "Otsu":
            self.global_threshold.setVisible(False)
            self.block_size.setVisible(False)
            self.c_value.setVisible(False)

        elif method == "Adaptive Threshold":
            self.global_threshold.setVisible(False)
            self.block_size.setVisible(True)
            self.c_value.setVisible(True)

        # Cập nhật lại label theo widget đang hiển thị
        self.refresh_parameter_labels(
            self.threshold_parameter_box
        )

    def refresh_parameter_labels(self, group):
        layout = group.layout()

        for i in range(layout.rowCount()):
            label_item = layout.itemAt(
                i, QFormLayout.LabelRole
            )

            field_item = layout.itemAt(
                i, QFormLayout.FieldRole
            )

            if label_item and field_item:
                widget = field_item.widget()

                if widget:
                    visible = widget.isVisible()

                    label = label_item.widget()

                    if label:
                        label.setVisible(visible)

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

        self.original_image = path

        pixmap = QPixmap(path)
        if pixmap.isNull():
            QMessageBox.warning(
                self,
                "Lỗi",
                "Không thể đọc ảnh đã chọn."
            )
            return

        self.threshold_original.label.setPixmap(
            pixmap.scaled(
                self.threshold_original.label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

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

        self.threshold_original.show_text(
            f"VIDEO ĐÃ CHỌN\n\n{path}"
        )

        self.original_video = path

    # CHẠY XỬ LÝ

    def run_threshold(self):

        if not self.original_image:
            QMessageBox.warning(
                self,
                "Thông báo",
                "Vui lòng mở ảnh trước khi chạy xử lý."
            )
            return

        try:
            # Đọc ảnh bằng QImage rồi chuyển sang ảnh xám NumPy.
            image = QImage(self.original_image)

            if image.isNull():
                raise ValueError("Không thể đọc ảnh.")

            image = image.convertToFormat(
                QImage.Format_Grayscale8
            )

            width = image.width()
            height = image.height()

            ptr = image.bits()
            img_gray = np.frombuffer(
                ptr,
                dtype=np.uint8,
                count=height * width
            ).reshape((height, width))

            method = self.threshold_method.currentText()

            if method == "Global Threshold":
                threshold = self.global_threshold.value()
                result = global_threshold(
                    img_gray,
                    threshold
                )

            elif method == "Otsu":
                threshold = otsu_threshold(img_gray)
                result = global_threshold(
                    img_gray,
                    threshold
                )

            elif method == "Adaptive Threshold":
                result = adaptive_threshold(
                    img_gray,
                    self.block_size.value(),
                    self.c_value.value()
                )

            else:
                raise ValueError(
                    "Phương pháp xử lý không hợp lệ."
                )

            # Chuyển kết quả NumPy về QImage để hiển thị.
            result_image = QImage(
                result.data,
                width,
                height,
                width,
                QImage.Format_Grayscale8
            ).copy()

            pixmap = QPixmap.fromImage(result_image)

            self.threshold_result.label.setPixmap(
                pixmap.scaled(
                    self.threshold_result.label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )

            if method == "Otsu":
                self.threshold_comment.setPlainText(
                    f"Otsu tự động chọn ngưỡng = {threshold}"
                )
            else:
                self.threshold_comment.setPlainText(
                    f"Đã xử lý bằng phương pháp: {method}"
                )

            self.result = result

        except Exception as e:
            QMessageBox.critical(
                self,
                "Lỗi xử lý",
                f"Không thể xử lý ảnh:\n{e}"
            )

    # LƯU KẾT QUẢ

    def save_threshold(self):

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Lưu kết quả",
            "",
            "PNG (*.png);;JPG (*.jpg)"
        )

        if path:
            QMessageBox.information(
                self,
                "Lưu kết quả",
                "Đã chọn vị trí lưu kết quả."
            )

