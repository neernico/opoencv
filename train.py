import cv2
import os
import numpy as np

# 设置数据集路径
dataset_path = 'D:\\work\\code\\project-opencv\\dataset'



# 初始化正面和侧面人脸检测器 (从本地路径加载)
face_detector_front = cv2.CascadeClassifier(os.path.join('haarcascade_frontalface_default.xml'))
face_detector_side = cv2.CascadeClassifier(os.path.join('haarcascade_profileface.xml'))

# 初始化识别器
recognizer = cv2.face.LBPHFaceRecognizer_create()

# 准备训练数据
def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)

    faces = []
    labels = []
    label_dict = {}
    current_label = 0

    for dir_name in dirs:
        if not os.path.isdir(os.path.join(data_folder_path, dir_name)):
            continue

        label_dict[current_label] = dir_name
        subject_dir_path = os.path.join(data_folder_path, dir_name)
        subject_images_names = os.listdir(subject_dir_path)

        for image_name in subject_images_names:
            if image_name.startswith('.'):
                continue
            image_path = os.path.join(subject_dir_path, image_name)
            image = cv2.imread(image_path)
            if image is None:
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 先检测正面人脸，如果检测不到再使用侧脸检测
            faces_rect = face_detector_front.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
            if len(faces_rect) == 0:
                faces_rect = face_detector_side.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

            # 如果检测到人脸，添加到训练数据
            for (x, y, w, h) in faces_rect:
                face = gray[y:y + h, x:x + w]
                faces.append(face)
                labels.append(current_label)
                break  # 假设每张图像只有一个人脸

        current_label += 1

    return faces, labels, label_dict


print("准备训练数据...")
faces, labels, label_dict = prepare_training_data(dataset_path)
print("训练数据准备完成")
print("总人脸样本数:", len(faces))
print("总标签数:", len(set(labels)))

# 训练识别器
recognizer.train(faces, np.array(labels))

# 保存识别器和标签字典
recognizer.save('face_recognizer.yml')
np.save('label_dict.npy', label_dict)

print("训练完成并保存识别器。")
