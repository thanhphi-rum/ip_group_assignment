import os
import shutil
import tempfile
import time

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QImage, QFont
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QComboBox, QSpinBox,
    QGroupBox, QVBoxLayout, QHBoxLayout, QFormLayout, QFileDialog,
    QMessageBox, QScrollArea, QSizePolicy
)

from algorithms.detect_edges import edges
from algorithms.contour import contour_detect
from algorithms.bounding_box import detect_bounding_boxes


class ImageBox(QGroupBox):
    def __init__(self, title):
        super().__init__(title)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 22, 14, 14)
        self. _current_pixmap = None

        self.label = QLabel("Chưa có ảnh\nNhấn 'Mở ảnh' để bắt đầu")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumSize(420, 330)
        # Dùng Ignored để QLabel co giãn mượt cả khi phóng to lẫn thu nhỏ cửa sổ
        self.label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.label.setStyleSheet(
            "QLabel { border: 2px dashed #CBD5E1; border-radius: 12px; "
            "background: #F8FAFC; color: #94A3B8; font-size: 15px; }"
        )
        layout.addWidget(self.label)

    def show_text(self, text):
        self._current_pixmap = None
        self.label.setPixmap(QPixmap())
        self.label.setText(text)

    def show_pixmap(self, pixmap):
        self._current_pixmap = pixmap
        self.label.setText("")
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self):
        if self._current_pixmap is not None and not self._current_pixmap.isNull():
            self.label.setPixmap(
                self._current_pixmap.scaled(
                    self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scaled_pixmap()


class VideoWorker(QThread):
    # Phát tín hiệu từng frame: (frame_goc, frame_ket_qua, stats, so_frame_hien_tai, tong_so_frame)
    frame_ready = Signal(object, object, object, int, int)
    finished_ok = Signal(str, object, object, int)
    failed = Signal(str)

    def __init__(self, video_path, get_params_fn):
        super().__init__()
        self.video_path = video_path
        self.get_params_fn = get_params_fn
        self._stop = False

    def stop(self):
        self._stop = True

    def _process_single_frame(self, frame):
        params = self.get_params_fn()
        method = params["method"]

        if method.startswith("Edge"):
            return edges(frame, params["low"], params["high"])

        if method == "Contour":
            bk = params["blur_k"]
            if bk % 2 == 0:
                bk += 1
            return contour_detect(
                frame,
                blur_k=bk,
                thresh=params["threshold"],
                morph_k=params["morph_k"],
                min_erea=params["min_area"],
            )

        return detect_bounding_boxes(
            frame, params["threshold"], params["min_area"]
        )

    def run(self):
        cap = None
        writer = None
        output_path = None
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise ValueError("Không thể mở tệp video.")

            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0 or np.isnan(fps):
                fps = 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            frame_delay_ms = max(1, int(1000.0 / fps))

            last_result = None
            last_stats = None
            frame_count = 0

            while not self._stop:
                t_start = time.perf_counter()
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                # Khởi tạo VideoWriter theo đúng kích thước thực tế của frame đầu tiên
                if writer is None:
                    h, w = frame.shape[:2]
                    fd, output_path = tempfile.mkstemp(suffix=".mp4")
                    os.close(fd)
                    writer = cv2.VideoWriter(
                        output_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (w, h),
                    )

                result, stats = self._process_single_frame(frame)

                # Đảm bảo ảnh ghi vào VideoWriter luôn có 3 kênh màu (BGR) và đúng kích thước
                if result.ndim == 2:
                    result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
                else:
                    result_bgr = result
                if result_bgr.shape[:2] != frame.shape[:2]:
                    result_bgr = cv2.resize(
                        result_bgr, (frame.shape[1], frame.shape[0])
                    )

                writer.write(result_bgr)
                frame_count += 1
                last_result = result.copy()
                last_stats = stats

                # Gửi frame về giao diện để hiển thị video chạy trực tiếp
                self.frame_ready.emit(
                    frame.copy(), last_result, stats, frame_count, total_frames
                )

                # Giữ nhịp phát video theo đúng FPS thực tế
                elapsed_ms = int((time.perf_counter() - t_start) * 1000)
                sleep_ms = frame_delay_ms - elapsed_ms
                if sleep_ms > 0:
                    self.msleep(sleep_ms)

            if frame_count == 0:
                raise ValueError("Video không có khung hình hợp lệ.")

            # Đóng writer trước khi gửi đường dẫn file kết quả
            if writer is not None:
                writer.release()
                writer = None

            path_to_send = output_path
            output_path = None
            self.finished_ok.emit(
                path_to_send, last_result, last_stats, frame_count
            )
        except Exception as e:
            self.failed.emit(str(e))
        finally:
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()
            if output_path and os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chủ đề 5: Project 2 - Phát hiện lỗi bề mặt")
        self.resize(1280, 900)
        self.original_image = None
        self.original_image_path = None
        self.original_video = None
        self.result = None
        self.video_result_path = None
        self.video_worker = None
        self.last_processed_suffix = None
        self._current_params = {}
        self.setup_style()
        self.setup_ui()

    def setup_style(self):
        self.setStyleSheet("""
            QMainWindow { background: #EEF2F7; }
            QWidget { font-family: 'Segoe UI', Arial, sans-serif; }

            QGroupBox {
                background: white;
                border: 1px solid #E2E8F0;
                border-radius: 12px;
                margin-top: 18px;
                padding: 18px 14px 14px 14px;
                font-size: 13px;
                font-weight: 600;
                color: #1E293B;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 14px;
                padding: 0 8px;
                color: #4F46E5;
            }

            QLabel { font-size: 14px; color: #334155; background: transparent; }

            QPushButton {
                min-height: 40px;
                padding: 6px 20px;
                border-radius: 8px;
                font-size: 14px;
                background: white;
                border: 1px solid #CBD5E1;
                color: #334155;
            }
            QPushButton:hover { background: #F1F5F9; border-color: #94A3B8; }
            QPushButton:pressed { background: #E2E8F0; }
            QPushButton:disabled { background: #F1F5F9; color: #94A3B8; }

            QPushButton#primaryButton {
                background: #4F46E5;
                color: white;
                border: 1px solid #4F46E5;
            }
            QPushButton#primaryButton:hover { background: #4338CA; border-color: #4338CA; }
            QPushButton#primaryButton:disabled {
                background: #A5B4FC; border-color: #A5B4FC; color: white;
            }

            QPushButton#dangerButton {
                background: #EF4444;
                color: white;
                border: 1px solid #EF4444;
            }
            QPushButton#dangerButton:hover { background: #DC2626; border-color: #DC2626; }

            QPushButton#successButton {
                background: #10B981;
                color: white;
                border: 1px solid #10B981;
            }
            QPushButton#successButton:hover { background: #059669; border-color: #059669; }

            QComboBox, QSpinBox {
                min-height: 36px;
                font-size: 14px;
                padding: 0 12px;
                border: 1px solid #CBD5E1;
                border-radius: 8px;
                background: white;
                color: #1E293B;
            }
            QComboBox:hover, QSpinBox:hover { border-color: #94A3B8; }
            QComboBox:focus, QSpinBox:focus { border-color: #4F46E5; }
            QComboBox QAbstractItemView {
                background: white;
                border: 1px solid #E2E8F0;
                selection-background-color: #EEF2FF;
                selection-color: #4F46E5;
                padding: 4px;
            }

            QScrollArea { background: transparent; border: none; }
            QScrollBar:vertical {
                background: #E2E8F0; width: 10px; border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #CBD5E1; border-radius: 5px; min-height: 30px;
            }
            QScrollBar::handle:vertical:hover { background: #94A3B8; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        """)

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(22, 16, 22, 16)
        root.setSpacing(6)

        title = QLabel("PROJECT 2: PHÁT HIỆN LỖI BỀ MẶT")
        title.setAlignment(Qt.AlignCenter)
        f = QFont()
        f.setPointSize(22)
        f.setBold(True)
        title.setFont(f)
        title.setStyleSheet("color: #1E293B; padding: 6px 0;")
        root.addWidget(title)

        subtitle = QLabel("NHÓM 12")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #64748B; font-size: 14px;")
        root.addWidget(subtitle)

        root.addWidget(self.create_detection_tab(), 1)

    def create_file_buttons(self):
        box = QGroupBox("Tệp và xử lý")
        layout = QHBoxLayout(box)
        layout.setContentsMargins(16, 24, 16, 16)
        layout.setSpacing(10)

        self.btn_image = QPushButton("Mở ảnh")
        self.btn_video = QPushButton("Mở video")
        self.process_btn = QPushButton("▶  Chạy xử lý")
        self.save_btn = QPushButton("Lưu kết quả")
        self.process_btn.setObjectName("primaryButton")
        self.save_btn.setObjectName("successButton")

        self.btn_image.clicked.connect(self.open_image)
        self.btn_video.clicked.connect(self.open_video)
        self.process_btn.clicked.connect(self.run_detection)
        self.save_btn.clicked.connect(self.save_result)

        layout.addWidget(self.btn_image)
        layout.addWidget(self.btn_video)
        layout.addStretch()
        layout.addWidget(self.process_btn)
        layout.addWidget(self.save_btn)
        return box

    def create_detection_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(12)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self.create_file_buttons())

        images = QHBoxLayout()
        images.setSpacing(12)
        self.original_box = ImageBox("ẢNH / VIDEO GỐC")
        self.result_box = ImageBox("KẾT QUẢ SAU KHI PHÁT HIỆN LỖI")
        images.addWidget(self.original_box, 1)
        images.addWidget(self.result_box, 1)
        layout.addLayout(images)

        settings = QHBoxLayout()
        settings.setSpacing(12)

        method_box = QGroupBox("Phương pháp phát hiện")
        method_layout = QVBoxLayout(method_box)
        method_layout.setContentsMargins(16, 24, 16, 16)
        method_layout.setSpacing(8)
        self.method = QComboBox()
        self.method.addItems(["Edge (Canny)", "Contour", "Bounding Box"])
        self.method.currentIndexChanged.connect(self.update_parameters)
        method_layout.addWidget(QLabel("Chọn kỹ thuật:"))
        method_layout.addWidget(self.method)
        method_layout.addStretch()

        self.parameter_box = QGroupBox("Tham số điều chỉnh")
        form = QFormLayout(self.parameter_box)
        form.setContentsMargins(16, 24, 16, 16)
        form.setSpacing(10)

        self.low_label = QLabel("Ngưỡng thấp (Low):")
        self.low = QSpinBox()
        self.low.setRange(0, 255)
        self.low.setValue(50)

        self.high_label = QLabel("Ngưỡng cao (High):")
        self.high = QSpinBox()
        self.high.setRange(0, 255)
        self.high.setValue(150)

        self.threshold_label = QLabel("Ngưỡng nhị phân:")
        self.threshold = QSpinBox()
        self.threshold.setRange(0, 255)
        self.threshold.setValue(100)

        self.area_label = QLabel("Diện tích lỗi tối thiểu:")
        self.min_area = QSpinBox()
        self.min_area.setRange(0, 1000000)
        self.min_area.setValue(30)

        self.blur_label = QLabel("Độ mờ (blur_k - số lẻ):")
        self.blur_k = QSpinBox()
        self.blur_k.setRange(1, 99)
        self.blur_k.setSingleStep(2)
        self.blur_k.setValue(5)

        self.morph_label = QLabel("Độ lọc nhiễu (morph_k):")
        self.morph_k = QSpinBox()
        self.morph_k.setRange(1, 99)
        self.morph_k.setValue(3)

        # Đồng bộ tham số mỗi khi người dùng thay đổi giá trị
        for spin in (self.low, self.high, self.threshold, self.min_area, self.blur_k, self.morph_k):
            spin.valueChanged.connect(self._sync_params)

        form.addRow(self.low_label, self.low)
        form.addRow(self.high_label, self.high)
        form.addRow(self.threshold_label, self.threshold)
        form.addRow(self.area_label, self.min_area)
        form.addRow(self.blur_label, self.blur_k)
        form.addRow(self.morph_label, self.morph_k)

        settings.addWidget(method_box, 1)
        settings.addWidget(self.parameter_box, 2)
        layout.addLayout(settings)

        info_box = QGroupBox("Kết quả phân tích")
        info_layout = QVBoxLayout(info_box)
        info_layout.setContentsMargins(16, 24, 16, 16)
        self.info = QLabel("Kết quả phân tích sẽ hiển thị tại đây.")
        self.info.setWordWrap(True)
        self.info.setStyleSheet(
            "QLabel { background: #F8FAFC; border: 1px solid #E2E8F0; "
            "border-radius: 8px; padding: 14px; color: #334155; font-size: 14px; }"
        )
        info_layout.addWidget(self.info)
        layout.addWidget(info_box)

        scroll.setWidget(content)
        self.update_parameters()
        return scroll

    def _sync_params(self):
        bk = self.blur_k.value()
        if bk % 2 == 0:
            bk += 1
        self._current_params = {
            "method": self.method.currentText(),
            "low": self.low.value(),
            "high": self.high.value(),
            "threshold": self.threshold.value(),
            "min_area": self.min_area.value(),
            "blur_k": bk,
            "morph_k": self.morph_k.value(),
        }

    def get_current_params(self):
        return dict(self._current_params)

    def update_parameters(self):
        method = self.method.currentText()
        is_edge = method.startswith("Edge")
        is_contour = (method == "Contour")

        self.low_label.setVisible(is_edge)
        self.low.setVisible(is_edge)
        self.high_label.setVisible(is_edge)
        self.high.setVisible(is_edge)

        self.threshold_label.setVisible(not is_edge)
        self.threshold.setVisible(not is_edge)
        self.area_label.setVisible(not is_edge)
        self.min_area.setVisible(not is_edge)

        self.blur_label.setVisible(is_contour)
        self.blur_k.setVisible(is_contour)
        self.morph_label.setVisible(is_contour)
        self.morph_k.setVisible(is_contour)

        self._sync_params()

    # ---- Quản lý luồng video và dọn file tạm ----

    def stop_video_worker(self):
        if self.video_worker is not None:
            if self.video_worker.isRunning():
                self.video_worker.stop()
                self.video_worker.wait(3000)
            self.video_worker = None
        self._set_video_running_ui(False)

    def cleanup_temp_video(self):
        if self.video_result_path and os.path.exists(self.video_result_path):
            try:
                os.remove(self.video_result_path)
            except OSError:
                pass
        self.video_result_path = None

    def _set_video_running_ui(self, running):
        if running:
            self.process_btn.setText("⏹  Dừng video")
            self.process_btn.setObjectName("dangerButton")
            self.save_btn.setEnabled(False)
        else:
            self.process_btn.setText("▶  Chạy xử lý")
            self.process_btn.setObjectName("primaryButton")
            self.save_btn.setEnabled(True)
        self.process_btn.style().unpolish(self.process_btn)
        self.process_btn.style().polish(self.process_btn)

    # ---- Đọc/ghi ảnh hỗ trợ tên file Unicode (Windows hay gặp lỗi) ----

    @staticmethod
    def imread(path):
        try:
            data = np.fromfile(path, dtype=np.uint8)
            if data.size == 0:
                return None
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        except OSError:
            return None

    @staticmethod
    def imwrite(path, image):
        ext = os.path.splitext(path)[1] or ".png"
        ok, buf = cv2.imencode(ext, image)
        if not ok:
            return False
        try:
            buf.tofile(path)
            return True
        except OSError:
            return False

    # ---- Mở file ----

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn ảnh", "", "Ảnh (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)"
        )
        if not path:
            return
        self.stop_video_worker()
        self.cleanup_temp_video()

        image = self.imread(path)
        if image is None:
            QMessageBox.warning(self, "Lỗi", "Không thể đọc ảnh đã chọn.")
            return
        self.original_image = image
        self.original_image_path = path
        self.original_video = None
        self.result = None
        self.last_processed_suffix = None
        self.original_box.show_pixmap(self.to_pixmap(image))
        self.result_box.show_text("Chưa xử lý")
        self.info.setText(
            f"Ảnh: {os.path.basename(path)} | Kích thước: {image.shape[1]} x {image.shape[0]}"
        )

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn video", "", "Video (*.mp4 *.avi *.mov *.mkv)"
        )
        if not path:
            return
        self.stop_video_worker()
        self.cleanup_temp_video()

        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        cap.release()
        if not ok or frame is None:
            QMessageBox.warning(self, "Lỗi", "Không thể đọc video đã chọn.")
            return
        self.original_video = path
        self.original_image = None
        self.original_image_path = None
        self.result = None
        self.last_processed_suffix = None
        self.original_box.show_pixmap(self.to_pixmap(frame))
        self.result_box.show_text("Video đã chọn - nhấn '▶ Chạy xử lý' để phát")
        self.info.setText(
            f"Video: {os.path.basename(path)} | Kích thước: {frame.shape[1]}x{frame.shape[0]} "
            f"| FPS: {fps:.1f} | Tổng số khung hình: {total}"
        )

    # ---- Xử lý ----

    def process_frame(self, frame):
        self._sync_params()
        params = self._current_params
        method = params["method"]

        if method.startswith("Edge"):
            return edges(frame, params["low"], params["high"])

        if method == "Contour":
            return contour_detect(
                frame,
                blur_k=params["blur_k"],
                thresh=params["threshold"],
                morph_k=params["morph_k"],
                min_erea=params["min_area"],
            )

        return detect_bounding_boxes(
            frame, params["threshold"], params["min_area"]
        )

    def run_detection(self):
        # Nếu video đang phát mà bấm nút thì dừng video
        if self.video_worker is not None and self.video_worker.isRunning():
            self.stop_video_worker()
            return

        if self.original_image is None and self.original_video is None:
            QMessageBox.warning(self, "Thông báo", "Vui lòng mở ảnh hoặc video trước.")
            return

        if self.original_image is not None:
            try:
                result, stats = self.process_frame(self.original_image)
                self.result = result
                self.last_processed_suffix = self.technique_suffix()
                self.result_box.show_pixmap(self.to_pixmap(result))
                self.info.setText(self.format_stats(stats))
            except Exception as e:
                QMessageBox.critical(self, "Lỗi xử lý", f"Không thể xử lý dữ liệu:\n{e}")
            return

        self.start_video_processing()

    def start_video_processing(self):
        self.stop_video_worker()
        self.cleanup_temp_video()
        self._sync_params()
        self.last_processed_suffix = self.technique_suffix()
        self._set_video_running_ui(True)
        self.info.setText("Đang khởi chạy xử lý video...")

        self.video_worker = VideoWorker(self.original_video, self.get_current_params)
        self.video_worker.frame_ready.connect(self.on_video_frame)
        self.video_worker.finished_ok.connect(self.on_video_finished)
        self.video_worker.failed.connect(self.on_video_failed)
        self.video_worker.finished.connect(self.on_video_thread_done)
        self.video_worker.start()

    def on_video_frame(self, orig_frame, result_frame, stats, frame_idx, total_frames):
        self.result = result_frame
        self.original_box.show_pixmap(self.to_pixmap(orig_frame))
        self.result_box.show_pixmap(self.to_pixmap(result_frame))
        progress = f"{frame_idx}/{total_frames}" if total_frames > 0 else f"{frame_idx}"
        self.info.setText(f"Đang phát video (Khung hình {progress}) | {self.format_stats(stats)}")

    def on_video_finished(self, output_path, last_frame, last_stats, frame_count):
        self.cleanup_temp_video()
        self.video_result_path = output_path
        self.result = last_frame
        if last_frame is not None:
            self.result_box.show_pixmap(self.to_pixmap(last_frame))
        stats_text = self.format_stats(last_stats) if last_stats is not None else ""
        self.info.setText(
            f"Đã phát và xử lý xong {frame_count} khung hình. {stats_text} "
            "| Bấm 'Lưu kết quả' để lưu video hoặc 'Chạy xử lý' để phát lại."
        )

    def on_video_failed(self, msg):
        QMessageBox.critical(self, "Lỗi xử lý", f"Không thể xử lý video:\n{msg}")
        self.info.setText("Xử lý video thất bại.")

    def on_video_thread_done(self):
        self._set_video_running_ui(False)
        self.video_worker = None

    def format_stats(self, stats):
        if isinstance(stats, int):
            return f"Kỹ thuật: Contour | Số vùng/đường bao đạt điều kiện: {stats}"
        if isinstance(stats, dict):
            if "edge_pixels" in stats:
                return f"Kỹ thuật: Canny Edge | Số điểm ảnh biên phát hiện được: {stats['edge_pixels']:,}"
            if "boxes" in stats:
                return f"Kỹ thuật: Bounding Box | Số khung lỗi phát hiện được: {stats['boxes']}"
        return f"Kết quả: {stats}"

    # ---- Lưu kết quả (tự đặt tên: tenanh_tenkythuat) ----

    def technique_suffix(self):
        method = self.method.currentText()
        if method.startswith("Edge"):
            return "canny"
        if method == "Contour":
            return "contour"
        return "bbox"

    @staticmethod
    def unique_path(path):
        if not os.path.exists(path):
            return path
        stem, ext = os.path.splitext(path)
        i = 1
        while os.path.exists(f"{stem}_{i}{ext}"):
            i += 1
        return f"{stem}_{i}{ext}"

    def save_result(self):
        suffix = getattr(self, "last_processed_suffix", None) or self.technique_suffix()

        # Lưu Video
        if self.video_result_path and os.path.exists(self.video_result_path):
            src = self.original_video
            if src and os.path.exists(src):
                stem = os.path.splitext(os.path.basename(src))[0]
                folder = os.path.dirname(os.path.abspath(src))
            else:
                stem = "video_ket_qua"
                folder = os.getcwd()

            default_path = os.path.join(folder, f"{stem}_{suffix}.mp4")
            out, _ = QFileDialog.getSaveFileName(
                self,
                "Chọn nơi lưu video kết quả",
                default_path,
                "Video MP4 (*.mp4);;Tất cả tệp (*.*)"
            )
            if not out:
                return
            if not out.lower().endswith(".mp4"):
                out += ".mp4"

            try:
                shutil.copy2(self.video_result_path, out)
            except OSError as e:
                QMessageBox.critical(self, "Lỗi", f"Không thể lưu video:\n{e}")
                return
            QMessageBox.information(self, "Lưu kết quả", f"Đã lưu video tại:\n{out}")
            return

        # Lưu Ảnh
        if self.result is None:
            QMessageBox.warning(self, "Thông báo", "Chưa có kết quả để lưu.")
            return

        src = self.original_image_path
        if src and os.path.exists(src):
            stem = os.path.splitext(os.path.basename(src))[0]
            folder = os.path.dirname(os.path.abspath(src))
        else:
            stem = "ket_qua"
            folder = os.getcwd()

        default_path = os.path.join(folder, f"{stem}_{suffix}.png")
        out, _ = QFileDialog.getSaveFileName(
            self,
            "Chọn nơi lưu ảnh kết quả",
            default_path,
            "Ảnh PNG (*.png);;Ảnh JPEG (*.jpg *.jpeg);;Ảnh BMP (*.bmp);;Ảnh TIFF (*.tif *.tiff)"
        )
        if not out:
            return
        if not os.path.splitext(out)[1]:
            out += ".png"

        if not self.imwrite(out, self.result):
            QMessageBox.critical(self, "Lỗi", "Không thể lưu ảnh kết quả.")
            return
        QMessageBox.information(self, "Lưu kết quả", f"Đã lưu ảnh tại:\n{out}")

    # ---- Hiển thị ----

    @staticmethod
    def to_pixmap(image):
        if image.ndim == 2:
            gray = np.ascontiguousarray(image)
            h, w = gray.shape
            qimage = QImage(
                gray.data, w, h, gray.strides[0], QImage.Format_Grayscale8
            ).copy()
        else:
            rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            h, w, _ = rgb.shape
            qimage = QImage(
                rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888
            ).copy()
        return QPixmap.fromImage(qimage)

    def closeEvent(self, event):
        self.stop_video_worker()
        self.cleanup_temp_video()
        event.accept()