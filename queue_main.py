#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   queue_main.py
@Time    :   2023/10/25 02:23:45
@Author  :   Li Ruilong
@Version :   1.0
@Contact :   liruilonger@gmail.com
@Desc    :   队列版本的 mtcnn 
"""

# here put the import lib


# here put the import lib
"""
pip install redis
"""


from globals import GlobalObject
import yaml_utils as Yaml
from functools import wraps
from PIL import Image

from redis_uits import RedisClient 

import pickle
import utils
import logging
import json
import os



logging.basicConfig(level=logging.INFO, )


global_object = GlobalObject()

config = Yaml.get_yaml_config(file_name="config/config.yaml")
queue_config = config['queue']



def url_detect_face(image_url,filename):
    """
    @Time    :   2023/09/18 01:41:00
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   通过照片 URL 检测
    """
    try:
        logging.info("处理的的图片路径为：{}".format(image_url))
        if utils.is_valid_url(image_url):
            body = utils.get_img_url_byte(image_url)
            #filename = os.path.basename(image_url)
        if filename == '':
            raise Exception("Upload failed, no files found ^_^")
    except Exception as e:
        logging.error(e)
        raise Exception(f"Upload failed, no files found ^_^{e}")

    try:
        # 验证图片完整性
        if utils.is_image_file(body):
            json_data = detect_face(body, filename)
            return json.loads(json_data)
        else:
            raise Exception(f"File {filename} uploaded successfully, parsing failed, image incomplete ^_^")
    except Exception as e:
        logging.error(e)
        raise  Exception(f"File {filename} uploaded successfully, parsing failed for unknown reason ^_^{e}")



def detect_face(body, filename):
    """
    @Time    :   2023/09/17 21:27:47
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   
    """
    faces = global_object.mtcnn.detect_face(body, filename)
    json_data = json.dumps(faces)
    return json_data


if __name__ == "__main__":
    image_key = queue_config['image_queue_key']
    face_key = queue_config['single_face_queue_key']
    rc = RedisClient()
    logging.info("redis 建立连接成功 ！！！{},{}".format(image_key,face_key))
    while True:
        try:
            logging.info("摄像头照片消费, 对应 redis 队列：{}".format(image_key))
            _ , image_data = rc.blpop(image_key,timeout=3) 
            print(image_data)
            if image_data is not None :
                
                image_data = image_data.decode('utf-8')
                
                image_url = image_data.split("@")[1]
                image_name = image_data.split("@")[0]
                
                # 判断是否结束标志
                if image_name  == 'Finish':
                    logging.info("队列中没有元素，进程结束")
                    break
                
                # 消费队列数据
                
                image_face = url_detect_face(image_url,image_name)
                logging.info("提取人脸信息，放入队列 {}".format(face_key))
                face_data = pickle.dumps(image_face)
                rc.rpush(face_key,face_data)
                
            else:
                logging.info("队列中暂时没有元素，循环等待 ^_^ ")
                continue
        except Exception as e :
            logging.error(f"{e}")
            continue  

        
        




    
    
