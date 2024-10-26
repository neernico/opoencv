import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class FaceRecognitionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("实时人脸识别系统")
        self.master.geometry("1000x800")
        self.master.resizable(False, False)

        # 加载正脸和侧脸 Haar Cascade 文件
        self.frontalface_cascade_path = os.path.join('haarcascades', 'D:\\conda\\envs\\opencv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
        self.profileface_cascade_path = os.path.join('haarcascades', 'D:\\conda\\envs\\opencv\\Lib\\site-packages\\cv2\\data\\haarcascade_profileface.xml')

        if not os.path.exists(self.frontalface_cascade_path) or not os.path.exists(self.profileface_cascade_path):
            messagebox.showerror("错误", "Haar Cascade 文件不存在。")
            self.master.destroy()
            return

        self.frontal_face_detector = cv2.CascadeClassifier(self.frontalface_cascade_path)
        self.profile_face_detector = cv2.CascadeClassifier(self.profileface_cascade_path)

        if self.frontal_face_detector.empty() or self.profile_face_detector.empty():
            messagebox.showerror("错误", "无法加载 Haar Cascade 文件。")
            self.master.destroy()
            return

        # 加载识别器模型
        self.model_path = 'face_recognizer.yml'
        if not os.path.exists(self.model_path):
            messagebox.showerror("错误", f"识别器模型文件不存在: {self.model_path}")
            self.master.destroy()
            return

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read(self.model_path)

        # 加载标签字典
        self.label_dict_path = 'label_dict.npy'
        if not os.path.exists(self.label_dict_path):
            messagebox.showerror("错误", f"标签字典文件不存在: {self.label_dict_path}")
            self.master.destroy()
            return

        self.label_dict = np.load(self.label_dict_path, allow_pickle=True).item()

        # 设置识别阈值
        self.threshold = 50  # 可根据需要调整

        # 当前模式（摄像头、图片、视频、无）
        self.current_mode = None

        # 创建界面元素
        self.create_widgets()

        # 标记摄像头线程是否正在运行
        self.camera_running = False
        self.cap = None

    def create_widgets(self):
        # 标题
        title_label = tk.Label(self.master, text="实时人脸识别系统", font=("Helvetica", 20))
        title_label.pack(pady=10)

        # 按钮框架
        button_frame = tk.Frame(self.master)
        button_frame.pack(pady=20)

        # 选择图片按钮
        select_image_btn = tk.Button(button_frame, text="选择图片进行识别", command=self.select_image, width=20, height=2)
        select_image_btn.grid(row=0, column=0, padx=10)

        # 启动摄像头识别按钮
        start_camera_btn = tk.Button(button_frame, text="启动摄像头识别", command=self.start_camera_recognition, width=20, height=2)
        start_camera_btn.grid(row=0, column=1, padx=10)

        # 选择视频文件按钮
        select_video_btn = tk.Button(button_frame, text="选择视频进行识别", command=self.select_video, width=20, height=2)
        select_video_btn.grid(row=0, column=2, padx=10)

        # 关闭程序按钮
        close_btn = tk.Button(button_frame, text="关闭程序", command=self.close_program, width=20, height=2)
        close_btn.grid(row=0, column=3, padx=10)

        # 显示区域
        self.image_label = tk.Label(self.master)
        self.image_label.pack(pady=10)

    def stop_current_mode(self):
        """ 停止当前的识别模式，无论是摄像头、图像或视频识别 """
        if self.current_mode == "camera":
            self.stop_camera()
        elif self.current_mode == "video":
            self.stop_video()

    def stop_camera(self):
        """ 停止摄像头识别 """
        if self.camera_running:
            self.camera_running = False
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()

    def stop_video(self):
        """ 停止视频识别 """
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

    def detect_faces(self, gray_frame):
        """ 同时检测正脸和侧脸 """
        frontal_faces = self.frontal_face_detector.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5)
        profile_faces = self.profile_face_detector.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5)

        return list(frontal_faces) + list(profile_faces)

    def select_image(self):
        """ 选择图片进行识别 """
        self.stop_current_mode()

        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
        )
        if not file_path:
            return

        self.current_mode = "image"
        image = cv2.imread(file_path)
        if image is None:
            messagebox.showerror("错误", "无法读取选择的图片。")
            return

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detect_faces(gray)

        if len(faces) == 0:
            messagebox.showinfo("信息", "未检测到人脸。")
            return

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            label, confidence = self.recognizer.predict(face)

            if confidence < self.threshold:
                name = self.label_dict.get(label, "Unknown")
                label_text = name  # 只显示名字
            else:
                label_text = "Unknown"

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image)

        # 增大显示区域到 800x600
        image_pil = image_pil.resize((800, 600), Image.LANCZOS)
        image_tk = ImageTk.PhotoImage(image_pil)
        self.image_label.config(image=image_tk)
        self.image_label.image = image_tk

    def start_camera_recognition(self):
        """ 启动摄像头进行识别 """
        self.stop_current_mode()
        self.current_mode = "camera"

        if self.camera_running:
            messagebox.showinfo("信息", "摄像头已在运行。")
            return

        self.camera_running = True
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            messagebox.showerror("错误", "无法打开摄像头。")
            self.camera_running = False
            return

        self.update_camera_frame()

    def update_camera_frame(self):
        """ 更新摄像头帧并显示到 Tkinter 窗口 """
        if not self.camera_running:
            return

        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("错误", "无法获取视频帧。")
            self.camera_running = False
            self.cap.release()
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detect_faces(gray)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            label, confidence = self.recognizer.predict(face)

            if confidence < self.threshold:
                name = self.label_dict.get(label, "Unknown")
                label_text = name  # 只显示名字
            else:
                label_text = "Unknown"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # 将摄像头帧大小调整为 800x600
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(frame)
        image_pil = image_pil.resize((800, 600), Image.LANCZOS)  # 调整尺寸
        image_tk = ImageTk.PhotoImage(image_pil)
        self.image_label.config(image=image_tk)
        self.image_label.image = image_tk

        self.master.after(10, self.update_camera_frame)

    def select_video(self):
        """ 选择视频文件进行识别 """
        self.stop_current_mode()
        file_path = filedialog.askopenfilename(
            title="选择视频",
            filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv")]
        )
        if not file_path:
            return

        self.current_mode = "video"
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            messagebox.showerror("错误", "无法打开视频文件。")
            self.current_mode = None
            return

        self.update_video_frame()

    def update_video_frame(self):
        """ 更新视频帧并显示到 Tkinter 窗口 """
        if self.current_mode != "video":
            return

        ret, frame = self.cap.read()
        if not ret:
            messagebox.showinfo("信息", "视频播放完毕。")
            self.stop_video()
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detect_faces(gray)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            label, confidence = self.recognizer.predict(face)

            if confidence < self.threshold:
                name = self.label_dict.get(label, "Unknown")
                label_text = name  # 修改为只显示类型（名字）
            else:
                label_text = "Unknown"

            # 绘制矩形框和标签
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # 将视频帧大小调整为 800x600
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(frame)
        image_pil = image_pil.resize((800, 600), Image.LANCZOS)  # 调整尺寸
        image_tk = ImageTk.PhotoImage(image_pil)
        self.image_label.config(image=image_tk)
        self.image_label.image = image_tk

        self.master.after(10, self.update_video_frame)

    def close_program(self):
        """ 关闭程序时停止所有进程 """
        self.stop_current_mode()
        self.master.quit()

def main():
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
