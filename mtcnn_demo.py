#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   mtcnn.py
@Time    :   2023/08/10 22:16:48
@Author  :   Li Ruilong
@Version :   1.0
@Contact :   liruilonger@gmail.com
@Desc    :   mtcnn demo 
"""

# here put the import lib
import warnings

import mtcnn
import cv2
import numpy as np
from imutils import paths
from mtcnn import MTCNN
import os
import utils
import yaml_utils as Yaml
from PIL import Image

from align_trans import warp_and_crop_face, get_reference_facial_points
from hopenet_demo import HopenetFace
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=UserWarning)


class MtcnnDetectFace:
    """
    @Time    :   2023/08/11 04:08:34
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   人脸检测工具类
    """

    def __init__(self, file_name="config/config.yaml") -> None:
        """
        @Time    :   2023/08/16 04:26:05
        @Author  :   liruilonger@gmail.com
        @Version :   1.0
        @Desc    :   初始化文件
                     Args:

                     Returns:
                       void
        """

        self.config = Yaml.get_yaml_config(file_name)
        mtcnn_config = self.config['mtcnn']['zero']

        hopenet_config = self.config['hopenet']['zero']

        self.min_face_size = mtcnn_config['min_face_size']
        self.scale_factor = mtcnn_config["scale_factor"]
        self.face_threshold = mtcnn_config['face_threshold']
        self.steps_threshold = mtcnn_config['steps_threshold']
        self.blur_threshold = mtcnn_config['blur_threshold']

        self.is_objectification = self.config['is_objectification']
        self.objectification_dir = self.config['objectification_dir']

        self.parse_dir = self.config['parse_dir']
        self.snapshot_path = hopenet_config['snapshot_path']
        self.yaw_threshold = hopenet_config['yaw_threshold']
        self.pitch_threshold = hopenet_config['pitch_threshold']
        self.roll_threshold = hopenet_config['roll_threshold']

        self.crop_size = (112, 112)
        self.refrence = get_reference_facial_points(
            default_square=self.crop_size[0] == self.crop_size[1])

    def build_model(self):
        """
        @Time    :   2023/08/10 22:39:47
        @Author  :   liruilonger@gmail.com
        @Version :   1.0
        @Desc    :   加载模型
                     Args:
                     min_face_size: 最小人脸尺寸
                     scale_factor: 影响因子

                     Returns:
                       void
        """
        face_detector = MTCNN(min_face_size=self.min_face_size, steps_threshold=self.steps_threshold,
                              scale_factor=self.scale_factor)
        face_hopenet = HopenetFace(self.snapshot_path)
        # 加载人脸检测模型
        self.face_detector = face_detector
        # 加载头部姿态判断模型
        self.face_hopenet = face_hopenet
        return self

    def detect_face(self, image,fname):
        """
        @Time    :   2023/08/14 03:17:22
        @Author  :   liruilonger@gmail.com
        @Version :   1.0
        @Desc    :   检测人脸
                     Args:

                     Returns:
                       void
        """

        img = utils.load_image_cvimg(image)
        img_PIL = utils.load_image_plimg(image)
        detected_face = None
        # mtcnn expects RGB but OpenCV read BGR
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections = self.face_detector.detect_faces(img_rgb)
        image_id = utils.get_uuid()
        resp = []
        if len(detections) > 0:

            for detection in detections:
                x, y, w, h = detection["box"]
                detected_face = img[int(y): int(y + h), int(x): int(x + w)]
                img_region = [x, y, w, h]

                # 置信度
                confidence = detection["confidence"]
                color = (255, 0, 0)
                if confidence <= self.face_threshold:
                    color = (255, 0, 255)

                if self.face_threshold > confidence:
                    print(
                        f"⚠️: {image_id} 中该置信度 {confidence}  未达到阈值 {self.face_threshold}，被弃用")
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, format(confidence, "0.5f"), (x - 5, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1,
                                cv2.LINE_4)
                    continue
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, format(confidence, "0.5f"), (x - 5, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1,
                            cv2.LINE_4)
                
                # 5 特征点
                facial5points = detection["keypoints"]
                # 用于人脸对齐，透视变换
                keypoints = list(
                    map(lambda xy: [xy[0], xy[1]], facial5points.values()))

                face_id = utils.get_uuid()
                # 用于人脸姿态检测
                detected_face_s = img_PIL.crop(
                    (int(x-30), int(y-30), int(x+w+30), int(y+h+30)))
                # 进行人脸对齐
                detected_face_align = self.alignment_procedure(
                    Image.fromarray(img_rgb), keypoints)
                # 进行模糊度检测
                blur_face = self.variance_of_laplacian(detected_face_align)

                if blur_face < self.blur_threshold:
                    cv2.putText(img, format(blur_face, "0.0f"), (x + w+2, y + h), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 1,
                                cv2.LINE_4)
                    print(
                        f"⚠️: {face_id} 中该模糊度 {blur_face} 未达到阈值 {self.blur_threshold}，被弃用")
                    continue
                cv2.putText(img, format(blur_face, "0.0f"), (x + w+2, y + h), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1,
                            cv2.LINE_4)
                # 进行头部姿态检测
                img_post, pitch, yaw, roll = self.face_hopenet.headPosEstimate(
                    detected_face_s)

                if abs(pitch) > self.pitch_threshold or abs(yaw) > self.yaw_threshold or abs(roll) > self.roll_threshold:
                    print(
                        f"⚠️: {face_id} 中该欧拉角 pitch： {pitch}， yaw：{yaw}, roll:{roll} 未达到阈值 {self.pitch_threshold},{self.yaw_threshold},{self.roll_threshold}，被弃用")
                    cv2.putText(img, format(pitch, "0.2f") + "/" + format(yaw, "0.2f") + "/" + format(roll, "0.2f"), (x + w, y + h//2), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 1,
                                cv2.LINE_4)
                    continue
                cv2.putText(img, format(pitch, "0.2f") + "/" + format(yaw, "0.2f") + "/" + format(roll, "0.2f"), (x + w, y + h//2), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1,
                                cv2.LINE_4)  

                # 是否持久化数据
                if self.is_objectification:
                    cv2.imwrite(self.objectification_dir + face_id[0:5] + "_" + format(
                        confidence, "0.5f")+"_native_image_.jpg", detected_face)
                    detected_face_s.save(
                        self.objectification_dir + face_id[0:5] + '_native_images_.jpg')
                    detected_face_align.save(
                        self.objectification_dir+face_id[0:5] + "_" + format(blur_face, "0.2f")+'.jpg')
                    cv2.imwrite(self.objectification_dir + face_id[0:5]+"p_"+format(pitch, "0.2f") + "_y_" + format(
                        yaw, "0.2f") + "_r_" + format(roll, "0.2f") + "_" + format(blur_face, "0.2f")+"_.jpg", img_post)
                resp.append({
                    "face_id": face_id,
                    "face_blur": blur_face,
                    "face_pose": {
                        "pitch": pitch,
                        "yaw": yaw,
                        "roll": roll
                    },
                    "face_confidence": confidence,
                    "face_coordinate": img_region,
                    "facie5points": facial5points,
                    "face_native_image_b64": utils.get_img_to_base64(detected_face),
                    "face_native_images_b64": utils.get_Image_to_base64(detected_face_s),
                    "face_align_images_b64": utils.get_Image_to_base64(detected_face_align),
                })
                
                    

        faces = {
            "image_id": image_id,
            "face_total": len(detections),
            "face_efficient_total_resp": len(resp),
            "resp": resp,
            "mark_image_face_b64": utils.get_img_to_base64(img),
        }
        if self.is_objectification:
            cv2.imwrite('./output/'+os.path.basename(fname), img)
        return faces

    def alignment_procedure(self, img, facial5points):
        """
        @Time    :   2023/08/14 03:16:43
        @Author  :   liruilonger@gmail.com
        @Version :   1.0
        @Desc    :   对图像进行人脸对齐,透视变化等操作
                     Args:

                     Returns:
                       void
        """
        # print(facial5points,type(img),type(self.refrence),type(self.crop_size))
        warped_face = warp_and_crop_face(
            np.array(img), facial5points, self.refrence, crop_size=self.crop_size)
        return Image.fromarray(warped_face)

    def variance_of_laplacian(self, image):
        """
        @Time    :   2023/07/25 01:57:44
        @Author  :   liruilonger@gmail.com
        @Version :   1.0
        @Desc    :   模糊度检测
                     Args:
                       image: openCV  中的 Image 对象 或者 PIL 中的  Image.Image 对象
                     Returns:
                       模糊度

        """
        if isinstance(image, Image.Image):
            numpy_image = np.array(image)
            cv2_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        else:
            cv2_image = image
        gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
        blur_face = cv2.Laplacian(gray, cv2.CV_64F).var()
        return blur_face


if __name__ == "__main__":
    mtcnn = MtcnnDetectFace()
    mtcnn.build_model()
    for ph in paths.list_images(mtcnn.parse_dir):
        print(f"处理照片：{ph}")
        faces = mtcnn.detect_face(ph)
        print(faces)
