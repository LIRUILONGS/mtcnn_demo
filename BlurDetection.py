#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   BlurDetection.py
@Time    :   2023/06/11 23:23:42
@Author  :   Li Ruilong
@Version :   1.0
@Contact :   liruilonger@gmail.com
@Desc    :   人脸模糊判断
实际测试中发现，阈值设置为 100 相对来说比较合适，当然如何数据集很大，可以考虑 提高阈值，当模糊度大于 1000 时，一般为较清晰图片，低于 100 时，图片模糊严重


"""

# here put the import lib

import cv2
from imutils import paths
import os


def calculate_blur(image):
    # 计算图像的拉普拉斯梯度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian



def is_blur_detection(img,Threshold=350):
    calculate_blur_ =  calculate_blur(img)
    if Threshold < calculate_blur_:
        return calculate_blur_,True
    else:
        return calculate_blur_,False

def one_blur_detection(path):
    """
    @Time    :   2023/06/13 02:17:56
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   单个图像输出图像清晰度指标
                 Args:
                   
                 Returns:
                   void
    """
    
    image = cv2.imread(path)
    print(calculate_blur(image))

if __name__ == "__main__":
    path = "W:\python_code\deepface\\temp\\cf\\cf_795acd36776904f9d96de77bbc9bee614.jpg"
    



    
    for path in paths.list_images("W:\python_code\deepface\\temp\\cf\\"):
        # 加载图像
        image = cv2.imread(path)
        # 计算图像清晰度指标
        blur,boo = is_blur_detection(image,350)
        if boo:
            composite_img = cv2.putText(image, str(blur), (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (255, 255, 255), 5, cv2.LINE_AA, False)
            cv2.imwrite(path+".png", composite_img)
            os.remove(path)
        else:
            print(f"图像清晰度指标:{blur}")


