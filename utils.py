#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   utils.py
@Time    :   2023/06/28 04:32:11
@Author  :   Li Ruilong
@Version :   1.0
@Contact :   liruilonger@gmail.com
@Desc    :   工具类
"""

# here put the import lib

import os
import pickle
import torch
import torch.distributed as dist
from urllib.parse import urlparse
import uuid
import requests
import base64
import numpy as np
import cv2
from io import BytesIO
import zipfile
import hashlib
from imutils import paths
import glob
from PIL import Image
import shutil
import io

def is_valid_url(url):
    """
    @Time    :   2023/05/29 21:49:19
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   url  字符串 校验
                 Args:
                   url
                 Returns:
                   booler
    """

    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def get_uuid():
    """
    @Time    :   2023/05/29 21:50:16
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   生成 UUID
                 Args:

                 Returns:
                   string
    """

    return str(uuid.uuid4()).replace('-', '')


def get_img_url_base64(url):
    """
    @Time    :   2023/05/29 21:50:42
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   图片 url 解析为 base64 编码
                 Args:
                   url
                 Returns:
                   base64_bytes
    """
    response = requests.get(url)
    image_bytes = response.content
    base64_bytes = base64.b64encode(image_bytes)
    return base64_bytes.decode('utf-8')

def get_img_url_byte(url):
    """
    @Time    :   2023/05/29 21:50:42
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   图片 url 解析为 字节
                 Args:
                   url
                 Returns:
                   base64_bytes
    """
    response = requests.get(url,verify=False)
    image_bytes = response.content
    return image_bytes


def get_img_url_Image(url):
    """
    @Time    :   2023/10/11 00:16:58
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   url 转化为 Image.image 对象
                 Args:
                   
                 Returns:
                   void
    """
    response = requests.get(url)
    image_bytes = response.content
    image = Image.open(BytesIO(image_bytes))
    return image

def get_image_to_base64(image_path):
    """
    @Time    :   2023/07/17 03:03:58
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   图片获取对应的 Baes64 编码
                 Args:

                 Returns:
                   void
    """

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode("utf-8")


def get_base64_to_img(base64_str):
    """
    @Time    :   2023/05/29 21:51:23
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   base64 编码转化为 opencv img 对象
                 Args:

                 Returns:
                   void
    """

    # 从 base64 编码的字符串中解码图像数据
    img_data = base64.b64decode(base64_str)
    # 将图像数据转换为 NumPy 数组
    nparr = np.frombuffer(img_data, np.uint8)
    # 解码图像数组
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def get_img_to_base64(img):
    """
    @Time    :   2023/05/29 21:54:26
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   opencv img 对象转化为 base64 编码
                 Args:

                 Returns:
                   void
    """
    img_b64 = base64.b64encode(cv2.imencode('.jpg', img)[1]).decode('utf-8')

    return img_b64


def get_b64s_and_make_to_zip(b64s_mark, img_id):
    # 创建一个名为 'images.zip' 的 zip 文件
    with zipfile.ZipFile(img_id+'_images.zip', 'w') as zip_file:
        # 遍历字典中的每个图像
        for img_b64, img_name in b64s_mark:
            # 将 base64 编码的数据解码为二进制数据
            img_data_binary = base64.b64decode(img_b64)
            # 将图像数据写入 zip 文件中
            zip_file.writestr(img_name+"_" + get_uuid() +
                              '.jpg', img_data_binary)

    # return send_file("..\\"+img_id+'_images.zip', as_attachment=True)


def build_img_text_marge(img_, text):
    """
    @Time    :   2023/06/01 05:29:09
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   生成文字图片拼接到 img 对象
                 Args:

                 Returns:
                   void
    """
    # 创建一个空白的图片
    img = np.zeros((500, 500, 3), dtype=np.uint8)

    # 设置字体和字号
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2

    # 在图片上写入文字
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness=2)
    text_x = (500 - text_size[0]) // 2
    text_y = (500 + text_size[1]) // 2
    cv2.putText(img, text, (text_x, text_y), font,
                font_scale, (0, 0, 0), thickness=2)
    montage_size = (300, 400)
    montages = cv2.build_montages([img_, img], montage_size, (1, 2))

    # 保存图片
    return montages


def image_to_base64(image):
    """
    @Time    :   2023/06/28 02:47:28
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   Image.image 对象转化为 base64 编码
                 Args:

                 Returns:
                   void
    """

    # 将图片转换为 base64 编码的字符串
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return base64_str


def get_base64_to_Image(image_base64):
    """
    @Time    :   2023/07/18 05:27:26
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   base64 编码 转化为  Image.image 对象
                 Args:

                 Returns:
                   void
    """

    image_data = base64.b64decode(image_base64)
    image = Image.open((image_data))
    return image


def get_base64_to_byte(image_base64):
    """
    @Time    :   2023/10/18 01:06:56
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   B64 编码转化为 字节
    """
    
    image_data = base64.b64decode(image_base64)
    return image_data

def get_file_md5(file_path):
    """
    @Time    :   2023/06/19 21:48:31
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   获取文件 MD5
                 Args:
                   file_path：str 文件路径
                 Returns:
                   MD5 对象的十六进制表示形式
    """

    with open(file_path, 'rb') as f:
        md5_obj = hashlib.md5()
        while True:
            data = f.read(4096)
            if not data:
                break
            md5_obj.update(data)
    return md5_obj.hexdigest()


def get_dir_md5(dir_path):
    """
    @Time    :   2023/06/19 23:26:20
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   获取目录图片文件的MD5
                 Args:
                   dir_path: 目录路径
                 Returns:
                   void
    """
    md5 = hashlib.md5()
    for img_path in paths.list_images(dir_path):
        md5.update(get_file_md5(img_path).encode())
    return md5.hexdigest()


def rm_suffix_file(dir_path, suffix="jpg"):
    """
    @Time    :   2023/06/27 23:18:51
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   删除指定后缀的文件
                 Args:

                 Returns:
                   void
    """
    if isinstance(dir_path, str):
        file_paths = glob.glob(os.path.join(dir_path, f"*.{suffix}"))
    else:
        file_paths = dir_path
    for file_path in file_paths:
        os.remove(file_path)


def mv_suffix_file(dir_path, destination_path='./temp', suffix="jpg"):
    """
    @Time    :   2023/06/27 23:18:51
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   删除指定后缀的文件
                 Args:

                 Returns:
                   void
    """
    if isinstance(dir_path, str):
        file_paths = glob.glob(os.path.join(dir_path, f"*.{suffix}"))
    else:
        file_paths = dir_path
    if not os.path.exists(destination_path):
        os.mkdir(destination_path)
    for file_path in file_paths:
        try:
            shutil.move(file_path, destination_path)
        except:
            pass
            try:
                os.remove(file_path)
            except:
                pass


def get_marge_image_to_base64(m1, m2, path, is_bash64=True):
    """
    @Time    :   2023/06/17 23:00:32
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   Image.Image 图片合并并转化为 Base64
    """
    if isinstance(m1, Image.Image):
        image1 = m1
    else:
        image1 = Image.open(m1)
    if isinstance(m2, Image.Image):
        image2 = m2
    else:
        image2 = Image.open(m2)

    # 获取第一张图片的大小
    width1, height1 = image1.size
    # 获取第二张图片的大小
    width2, height2 = image2.size
    # 创建一个新的画布，大小为两张图片的宽度之和和高度的最大值
    new_image = Image.new("RGB", (width1 + width2, max(height1, height2)))
    # 将第一张图片粘贴到画布的左侧
    new_image.paste(image1, (0, 0))
    # 将第二张图片粘贴到画布的右侧
    new_image.paste(image2, (width1, 0))
    if is_bash64:
        return get_Image_to_base64(new_image)
    else:
        new_image.save(path+os.path.basename(m1))


def get_Image_to_base64(image_):
    """
    @Time    :   2023/07/03 06:24:15
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   Image.Image 图片对象转化为 Base64
                 Args:

                 Returns:
                   void
    """
    if isinstance(image_, Image.Image):
        image = image_
    else:
        image = Image.open(image_)

    image_stream = BytesIO()
    image.save(image_stream, format='PNG')
    image_stream.seek(0)
    base64_data = base64.b64encode(image_stream.read()).decode('utf-8')
    return base64_data


def get_Image_size(image_):
    """
    @Time    :   2023/07/11 07:27:53
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   获取 Image.Image 对象大小
                 Args:

                 Returns:
                   void
    """
    if isinstance(image_, Image.Image):
        image = image_
    else:
        image = Image.open(image_)
    image_stream = BytesIO()
    image.save(image_stream, format='JPEG')
    return float("{:.2f}".format(len(image_stream.getvalue()) / 1024))


def get_image_md5_from_base64(base64_str):
    """
    @Time    :   2023/07/19 00:34:11
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   图像的  b64 编码获取 MD5 值
                 Args:

                 Returns:
                   void
    """

    # 解码 Base64 编码为二进制数据
    image_data = base64.b64decode(base64_str)

    # 计算二进制数据的 MD5 值
    md5_hash = hashlib.md5()
    md5_hash.update(image_data)
    md5_value = md5_hash.hexdigest()

    return md5_value


def save_image_from_base64(b64_str, op, fn):
    """
    @Time    :   2023/07/04 00:01:36
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   b64 保存为图片
                 Args:

                 Returns:
                   void
    """
    print(op, fn)
    image_data = base64.b64decode(b64_str)
    image = Image.open(BytesIO(image_data))

    image.save(os.path.join(op, os.path.basename(fn)))


def featuresa_and_featuresb(featuresa, featuresb, threshold=0.45):
    """
    @Time    :   2023/07/24 06:51:51
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   多个特征的相似度计算
                 Args:

                 Returns:
                   void
    """

    featuresa.append(featuresb)
    X = torch.cat(featuresa, dim=0)
    boo = False
    G = X[-1] @ X.T
    n = G.shape[0] - 1
    print(G)
    for i, score in enumerate(G):
        if i == n:
            break
        if score >= threshold:
            boo = True

    return boo


def featuresa_and_featuresb_all(featuresa, featuresb, threshold=0.45):
    """
    @Time    :   2023/07/24 06:51:51
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   多个特征的相似度计算
                 Args:

                 Returns:
                   void
    """

    for a in featuresa:
        for b in featuresb:
            if findCosineDistance_CPU(a, b) >= threshold:
                return True

    return False


def findCosineDistance_CPU(source_representation, test_representation):
    """
    @Time    :   2023/06/16 12:19:27
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   计算两个向量之间的余弦相似度得分，CPU 版本
                 Args:
                 Returns:
                   void
    """
    import torch.nn.functional as F
    return F.cosine_similarity(source_representation, test_representation)


def variance_of_laplacian(image):
    """
    @Time    :   2023/07/25 01:57:44
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   模糊度检测
                 Args:

                 Returns:
                   void
    """
    numpy_image = np.array(image)
    cv2_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def coordinate_rectangles(center, width=80, height=80):
    """
    @Time    :   2023/08/10 02:29:21
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   根据中心点绘制人脸矩形框
                 Args:

                 Returns:
                   void
    """

    x1 = int(center[0] - width / 2)
    y1 = int(center[1] - height / 2)
    x2 = int(center[0] + width / 2)
    y2 = int(center[1] + height / 2)
    return x1, y1, x2, y2


def load_image_cvimg(img):
    """
    @Time    :   2023/09/14 22:49:44
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   加载图片
    """

    exact_image = False
    base64_img = False
    url_img = False
    bytes_image = False

    if type(img).__module__ == np.__name__:
        exact_image = True
    elif isinstance(img, bytes):
        bytes_image = True

    elif img.startswith("data:image/"):
        base64_img = True

    elif img.startswith("http"):
        url_img = True

    # ---------------------------

    if base64_img is True:
        img = loadBase64Img(img)

    elif url_img is True:
        img = np.array(Image.open(requests.get(
            img, stream=True, timeout=60).raw).convert("RGB"))
    elif bytes_image is True:
        image_array = np.frombuffer(img, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    elif exact_image is not True:  # image path passed as input
        if os.path.isfile(img) is not True:
            raise ValueError(f"Confirm that {img} exists")

        img = cv2.imread(img)

    return img


def load_image_plimg(img):
    """
    @Time    :   2023/09/14 22:49:44
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   加载图片
    """

    base64_img = False
    url_img = False
    bytes_image = False

    if isinstance(img, bytes):
        bytes_image = True
    elif img.startswith("data:image/"):
        base64_img = True

    elif img.startswith("http"):
        url_img = True

    # ---------------------------

    if base64_img is True:
        # 提取Base64编码的图像数据部分（不包含"data:image/png;base64,"前缀）
        image_data = img.split(",")[1]
        # 解码Base64图像数据
        decoded_img = base64.b64decode(image_data)
        # 创建BytesIO对象
        image_stream = io.BytesIO(decoded_img)
        # 打开PIL Image对象
        img = Image.open(image_stream)

    elif url_img is True:
        response = requests.get(img)
        image_data = response.content
        image_stream = io.BytesIO(image_data)
        img = Image.open(image_stream)

    elif bytes_image is True:
        image_stream = io.BytesIO(img)
        img = Image.open(image_stream)

    return img

def loadBase64Img(uri):
    """
    @Time    :   2023/09/14 23:07:06
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   64 编码转 Image

    """
    
    encoded_data = uri.split(",")[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def is_image_file(image_bytes):
    """
    @Time    :   2023/09/14 23:06:52
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   判断是否为文件文件
    """
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            img.verify()  # 验证图像文件的完整性
        return True
    except (IOError, SyntaxError):
        return False




def is_monochrome(image_path, threshold=10):
    """
    @Time    :   2023/09/26 04:25:52
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   
    """
    
    image = Image.open(image_path)
    image = image.convert('L')  # 将图像转换为灰度图像

    # 获取图片的所有像素值
    pixels = image.getdata()
    # 获取图像的直方图
    histogram = image.histogram()

    # 计算直方图的标准差
    std_dev = sum((value - (sum(histogram) / len(histogram)))**2 for value in histogram) / len(histogram)
    print(std_dev)
    # 判断标准差是否小于阈值
    if std_dev <= threshold:
        return True
    else:
        return False



def get_color_ratio(image_path,quantity_threshold='35000',occupancy_threshold='0.006092959104938272'):
    """
    @Time    :   2023/10/06 22:13:51
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   判断图片颜色是否为单一颜色
    Occupancy threshold: 颜色种类阈值
    quantity_threshold： 单一颜色占比阈值



    """
    
    # 打开图像
    image = Image.open(image_path)
    # 转换为RGB模式
    image_rgb = image.convert("RGB")
    # 获取图像尺寸
    width, height = image_rgb.size

    # 统计颜色频次
    color_counts = {}
    total_pixels = width * height

    for y in range(height):
        for x in range(width):
            # 获取像素颜色
            pixel = image_rgb.getpixel((x, y))
            # 统计颜色频次
            if pixel in color_counts:
                color_counts[pixel] += 1
            else:
                color_counts[pixel] = 1

    # 计算颜色占比
    color_ratios = {}
    for color, count in color_counts.items():
        ratio = count / total_pixels
        color_ratios[color] = ratio
    #if  len(color_ratios) < quantity_threshold:
    #    return True
    #max_color =  max(zip(result.values(),result.keys()))
    #first_item = next(iter(max_color.items()))
    #value = first_item[1]
#
    #if  occupancy_threshold < first_item :

    return color_ratios


if __name__ == "__main__":
    # date =  get_Image_size("C:\\Users\\liruilong\\Documents\\GitHub\\AdaFace_demo\\res\\wj.jpg")
    # print(date )
    #for path in paths.list_images("./res/liruilong"):
    #    print(get_file_md5(path))
    

    # 调用函数判断图片是否为单色或接近单色图片
    img_path = "z:\\1111.jpg" #"z:\\20230926043918.png"
    result = get_color_ratio(img_path)
    print(result)
    print(max(zip(result.values(),result.keys())))
    print(len(result))

       

