import sys
import os
import cv2
import numpy as np
import mediapipe as mp

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QImage, QPixmap, QFont, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, 
    QHBoxLayout, QVBoxLayout, QGridLayout, QPushButton, 
    QLabel, QGroupBox
)

# --- 视觉任务工作线程 ---
class VisionWorker(QThread):
    frame_signal = Signal(QImage)
    status_signal = Signal(str, str)

    def __init__(self, task_type):
        super().__init__()
        self.is_running = True
        self.task_type = task_type

    def run(self):
        cap = cv2.VideoCapture(0)
        
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        segmenter_context = None
        detector = None

        try:
            if self.task_type == 'hands':
                mp_hands = mp.solutions.hands
                detector = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
            elif self.task_type == 'pose':
                mp_pose = mp.solutions.pose
                detector = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            elif self.task_type == 'segmentation':
                mp_seg = mp.solutions.selfie_segmentation
                detector = mp_seg.SelfieSegmentation(model_selection=1)
                bg_image = None
            elif self.task_type == 'objectron':
                mp_obj = mp.solutions.objectron
                detector = mp_obj.Objectron(static_image_mode=False, max_num_objects=5, min_detection_confidence=0.5, min_tracking_confidence=0.99, model_name='Shoe')
            elif self.task_type == 'hair':
                model_path = r"C:\Users\17244\Desktop\python\day06\hair_segmenter.tflite"

                if not os.path.exists(model_path):
                    self.status_signal.emit(f"错误: 未找到模型文件\n请检查路径: {model_path}", "red")
                    cap.release()
                    return

                with open(model_path, "rb") as f:
                    model_buffer = f.read()

                BaseOptions = mp.tasks.BaseOptions
                ImageSegmenter = mp.tasks.vision.ImageSegmenter
                ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
                VisionRunningMode = mp.tasks.vision.RunningMode

                options = ImageSegmenterOptions(
                    base_options=BaseOptions(model_asset_buffer=model_buffer, delegate=BaseOptions.Delegate.CPU),
                    running_mode=VisionRunningMode.IMAGE,
                    output_category_mask=True
                )
                segmenter_context = ImageSegmenter.create_from_options(options)
                detector = segmenter_context.__enter__()

            elif self.task_type == 'face_mesh':
                mp_face_mesh = mp.solutions.face_mesh
                detector = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
            elif self.task_type == 'face_mesh_full':
                mp_face_mesh = mp.solutions.face_mesh
                detector = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
            elif self.task_type == 'face_detection':
                mp_face_detection = mp.solutions.face_detection
                detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        except Exception as e:
            self.status_signal.emit(f"模型初始化异常: {str(e)}", "red")
            cap.release()
            return

        self.status_signal.emit(f"正在运行: {self.task_type.upper()} 监测", "green")

        while self.is_running and cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            h, w, _ = image.shape
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.task_type == 'hands':
                image.flags.writeable = False
                results = detector.process(image_rgb)
                image.flags.writeable = True
                image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                image = cv2.flip(image, 1)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            elif self.task_type == 'pose':
                image.flags.writeable = False
                results = detector.process(image_rgb)
                image.flags.writeable = True
                image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                image = cv2.flip(image, 1)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            elif self.task_type == 'segmentation':
                image.flags.writeable = False
                results = detector.process(image_rgb)
                image.flags.writeable = True
                image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
                if bg_image is None:
                    bg_image = np.zeros(image.shape, dtype=np.uint8)
                    bg_image[:] = (192, 192, 192)
                image = np.where(condition, image, bg_image)
                image = cv2.flip(image, 1)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            elif self.task_type == 'objectron':
                image.flags.writeable = False
                results = detector.process(image_rgb)
                image.flags.writeable = True
                image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                if results.detected_objects:
                    for obj in results.detected_objects:
                        mp_drawing.draw_landmarks(image, obj.landmarks_2d, mp_obj.BOX_CONNECTIONS)
                        mp_drawing.draw_axis(image, obj.rotation, obj.translation)
                image = cv2.flip(image, 1)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            elif self.task_type == 'hair':
                try:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                    segmentation_result = detector.segment(mp_image)
                    category_mask = segmentation_result.category_mask.numpy_view()
                    hair_mask_bool = (category_mask == 1)

                    colored_overlay = image.copy()
                    colored_overlay[hair_mask_bool] = [255, 0, 180] 

                    alpha = 0.4
                    result_frame = cv2.addWeighted(colored_overlay, alpha, image, 1 - alpha, 0)
                    image = cv2.flip(result_frame, 1)
                except Exception:
                    image = cv2.flip(image, 1)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            elif self.task_type == 'face_mesh':
                results = detector.process(image_rgb)
                image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        left_pupil = face_landmarks.landmark[468]
                        right_pupil = face_landmarks.landmark[473]
                        cx_left, cy_left = int(left_pupil.x * w), int(left_pupil.y * h)
                        cx_right, cy_right = int(right_pupil.x * w), int(right_pupil.y * h)
                        cv2.circle(image, (cx_left, cy_left), 4, (0, 0, 255), -1)
                        cv2.circle(image, (cx_right, cy_right), 4, (0, 0, 255), -1)
                image = cv2.flip(image, 1)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            elif self.task_type == 'face_mesh_full':
                image.flags.writeable = False
                results = detector.process(image_rgb)
                image.flags.writeable = True
                image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # 【优化】只绘制面部轮廓线条（FACEMESH_CONTOURS），不再渲染密集的网格面片
                        mp_drawing.draw_landmarks(
                            image=image, landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                        )
                image = cv2.flip(image, 1)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            elif self.task_type == 'face_detection':
                image.flags.writeable = False
                results = detector.process(image_rgb)
                image.flags.writeable = True
                image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                if results.detections:
                    for detection in results.detections:
                        mp_drawing.draw_detection(image, detection)
                image = cv2.flip(image, 1)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            h_img, w_img, ch = rgb_image.shape
            qt_image = QImage(rgb_image.data, w_img, h_img, ch * w_img, QImage.Format_RGB888)
            self.frame_signal.emit(qt_image)

        cap.release()
        if segmenter_context:
            try:
                segmenter_context.__exit__(None, None, None)
            except Exception:
                pass
        elif detector and hasattr(detector, 'close'):
            try:
                detector.close()
            except Exception:
                pass

    def stop(self):
        self.is_running = False
        self.wait()


# --- 主控制窗口 ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MediaPipe 视觉处理工作台")
        self.resize(780, 600)
        self.worker = None

        self.setStyleSheet("""
            QMainWindow { background-color: #f5f6f8; }
            QLabel { color: #333333; font-family: "Microsoft YaHei"; }
            QPushButton { 
                background-color: #ffffff; 
                color: #333333; 
                border: 1px solid #dcdde1; 
                border-radius: 4px; 
                font-size: 11px; 
                padding: 4px 6px; 
            }
            QPushButton:hover { background-color: #f1f2f6; border-color: #c8d6e5; }
            QPushButton:pressed { background-color: #dfe4ea; }
            QGroupBox { 
                border: 1px solid #dcdde1; 
                border-radius: 6px; 
                margin-top: 6px; 
                color: #2f3640; 
                font-weight: bold; 
                font-size: 11px;
                background-color: #ffffff;
            }
        """)

        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(6)

        # 1. 顶部控制与功能区
        top_layout = QHBoxLayout()
        top_layout.setSpacing(6)

        btn_group = QGroupBox("功能选择")
        btn_grid = QGridLayout()
        btn_grid.setContentsMargins(6, 10, 6, 6)
        btn_grid.setHorizontalSpacing(4)
        btn_grid.setVerticalSpacing(4)
        
        self.btn_hands = QPushButton("手势识别 (Hands)")
        self.btn_hands.clicked.connect(lambda: self.switch_task('hands'))
        btn_grid.addWidget(self.btn_hands, 0, 0)

        self.btn_pose = QPushButton("全身骨骼 (Pose)")
        self.btn_pose.clicked.connect(lambda: self.switch_task('pose'))
        btn_grid.addWidget(self.btn_pose, 0, 1)

        self.btn_seg = QPushButton("背景分割 (Seg)")
        self.btn_seg.clicked.connect(lambda: self.switch_task('segmentation'))
        btn_grid.addWidget(self.btn_seg, 0, 2)

        self.btn_obj = QPushButton("3D物体 (Objectron)")
        self.btn_obj.clicked.connect(lambda: self.switch_task('objectron'))
        btn_grid.addWidget(self.btn_obj, 0, 3)

        self.btn_hair = QPushButton("头发分割 (Hair)")
        self.btn_hair.clicked.connect(lambda: self.switch_task('hair'))
        btn_grid.addWidget(self.btn_hair, 1, 0)

        self.btn_face_mesh = QPushButton("瞳孔追踪 (Mesh)")
        self.btn_face_mesh.clicked.connect(lambda: self.switch_task('face_mesh'))
        btn_grid.addWidget(self.btn_face_mesh, 1, 1)

        self.btn_face_mesh_full = QPushButton("高精度轮廓 (Full)")
        self.btn_face_mesh_full.clicked.connect(lambda: self.switch_task('face_mesh_full'))
        btn_grid.addWidget(self.btn_face_mesh_full, 1, 2)

        self.btn_face_det = QPushButton("人脸检测 (Det)")
        self.btn_face_det.clicked.connect(lambda: self.switch_task('face_detection'))
        btn_grid.addWidget(self.btn_face_det, 1, 3)

        btn_group.setLayout(btn_grid)
        top_layout.addWidget(btn_group, 3)

        right_control_layout = QVBoxLayout()
        right_control_layout.setSpacing(4)
        
        status_group = QGroupBox("运行状态")
        status_layout = QVBoxLayout()
        status_layout.setContentsMargins(6, 10, 6, 6)
        self.status_label = QLabel("请选择功能启动...")
        self.status_label.setFont(QFont("Microsoft YaHei", 9))
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)
        status_group.setLayout(status_layout)
        right_control_layout.addWidget(status_group)

        self.btn_stop = QPushButton("关闭当前任务")
        self.btn_stop.clicked.connect(self.stop_current_task)
        self.btn_stop.setStyleSheet("background-color: #fff5f5; color: #c0392b; border: 1px solid #fab1a0; font-weight: bold;")
        right_control_layout.addWidget(self.btn_stop)

        top_layout.addLayout(right_control_layout, 1)
        main_layout.addLayout(top_layout)

        # 2. 下方视频显示视窗
        self.video_label = QLabel("视窗未激活（请点击上方按钮开始）")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            background-color: #2f3640; 
            border: 1px solid #dcdde1; 
            border-radius: 6px;
            color: #f5f6f8;
            font-size: 12px;
        """)
        self.video_label.setMinimumSize(560, 360)
        main_layout.addWidget(self.video_label, 1)

        self.setCentralWidget(central_widget)

    def switch_task(self, task_type):
        if self.worker:
            self.worker.stop()
            self.worker = None

        self.worker = VisionWorker(task_type)
        self.worker.frame_signal.connect(self.update_image)
        self.worker.status_signal.connect(self.update_status)
        self.worker.start()

    def stop_current_task(self):
        if self.worker:
            try:
                self.worker.frame_signal.disconnect()
                self.worker.status_signal.disconnect()
            except Exception:
                pass
            
            self.worker.stop()
            self.worker = None
        
        self.video_label.clear()
        self.video_label.setText("视窗已关闭")
        self.video_label.setStyleSheet("""
            background-color: #2f3640; 
            border: 1px solid #dcdde1; 
            border-radius: 6px;
            color: #f5f6f8;
            font-size: 12px;
        """)
        self.update_status("当前没有运行中的任务。", "green")

    def update_image(self, qt_image):
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def update_status(self, text, color_type):
        self.status_label.setText(text)
        colors = {
            "green": "#27ae60", 
            "red": "#c0392b", 
            "yellow": "#d35400"
        }
        active_color = colors.get(color_type, "#333333")
        
        palette = self.status_label.palette()
        palette.setColor(self.status_label.foregroundRole(), QColor(active_color))
        self.status_label.setPalette(palette)
        
        current_font = self.status_label.font()
        current_font.setBold(True)
        self.status_label.setFont(current_font)

    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())