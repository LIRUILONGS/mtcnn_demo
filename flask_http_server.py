#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   flask_http_server.py
@Time    :   2023/09/17 21:45:06
@Author  :   Li Ruilong
@Version :   1.0
@Contact :   liruilonger@gmail.com
@Desc    :   flask 版本的 httpd 服务
"""

# here put the import lib
"""
pip install flask
"""



from flask import  Flask  # 导入Flask类
from flask import Flask, render_template, request,jsonify,Response
from PIL import Image
from flask import make_response
from flask import current_app
import utils
import logging
import json
import os

from functools import wraps
from flask import abort
from concurrent.futures import ThreadPoolExecutor
from mtcnn_demo import MtcnnDetectFace
import yaml_utils as Yaml




app = Flask(__name__)  # 实例化并命名为app实例
# 线程池执行器
executor = ThreadPoolExecutor()
logging.basicConfig(level=logging.INFO, )

config = Yaml.get_yaml_config(file_name="config/config.yaml")
flask_config = config['flask']



class MyContextObject:
    """
    @Time    :   2023/09/14 22:44:35
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   全局上下文对象
    """

    def __init__(self, mtcnn, ready_mark=False):
        self.mtcnn = mtcnn
        self.ready_mark = ready_mark

def init_obj():
    """
    @Time    :   2023/09/18 01:33:26
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   上下文对象处理
    """

    mtcnn = MtcnnDetectFace()
    logging.info('🚀🚀🚀🚀 人脸检测相关模型加载')
    mtcnn.build_model()
    logging.info('🚀🚀🚀🚀 构建上下文对象')
    my_context = MyContextObject(mtcnn, ready_mark=True)
    current_app.my_context = my_context
    logging.info('🚀🚀🚀🚀🚀 \033[32m服务启动成功\033[0m')

with app.app_context():
       init_obj()


# 定义装饰器函数，用于验证 Token
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token is missing'}), 401

        valid_tokens = flask_config["token"]
        # 如果 Token 无效，返回未授权的响应
        if token not in valid_tokens:
             return jsonify({'message': 'Token is invalid'}), 401

        # 如果 Token 有效，继续处理请求
        return f(*args, **kwargs)

    return decorated



@app.route("/")
def index():
    """
    @Time    :   2023/09/17 22:40:15
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   欢迎页
    """

    return {'result': "Hello, face"}


@app.route("/livez")
def livez():
    """
    @Time    :   2023/09/17 22:44:43
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   服务存活探针接口
    """
    return {'result': " live  ^_^"}


@app.route("/token")
def token():
    return jsonify({
            "code": 200,
            "message": "获取 token",
            "token": flask_config["token"],
        })


@app.route("/readyz")
def readyz():
    """
    @Time    :   2023/09/17 22:46:20
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   服务就绪探针接口
    """
    context = app.my_context
    if context.ready_mark:
        return {'result': " ready ^_^ "} 
    else:
        abort(503, "Service Unavailable")
    



@app.route("/upload", methods=["POST"])
@token_required
def upload():
    """
    @Time    :   2023/09/18 01:41:00
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   单文件上传检测
    """
    try:
        file = request.files['image']
        if file.filename == '':
            abort(404, "Upload failed, no files found ^_^")
        filename = file.filename
        body = file.read()
    except Exception as e:
        abort(404, f"Upload failed, no files found ^_^{e}")
    try:
        # 验证图片完整性
        if utils.is_image_file(body):
            json_data = detect_face(body, filename)
            return  Response(json_data, mimetype='application/json')
        else:
            abort(
                400, f"File {filename} uploaded successfully, parsing failed, image incomplete ^_^")
    except Exception as e:
        abort(
            500, f"File {filename} uploaded successfully, parsing failed for unknown reason ^_^{e}")



@app.route("/uploads", methods=["POST"])
@token_required
def uploads():
    """
    @Time    :   2023/10/10 02:22:33
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   多文件上传检测
    """
    try:
        if 'image' not in request.files:
            abort(400, "Upload failed, no files found ^_^")
        files = request.files.getlist('image')
        
    except Exception as e:
        abort(400, f"Upload failed, no files found ^_^{e}")
        
    try:
        # 验证图片完整性
        face_imges = []
        for file in files:
            if file:
                filename = file.filename
                body = file.read()
                if utils.is_image_file(body):
                    json_data = detect_face(body, filename)
                    face_imges.append(json_data)
        response = Response(face_imges, mimetype='application/json')
        return  response            
    except Exception as e:

        abort(400, f"File {filename} uploaded successfully, parsing failed for unknown reason ^_^ {e}")
   




def detect_face(body, filename):
    """
    @Time    :   2023/09/17 21:27:47
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   
    """
    context = current_app.my_context
    faces =  context.mtcnn.detect_face(body, filename)
    json_data = json.dumps(faces)
    return json_data


if __name__ == "__main__":
    app.run(port=flask_config['port'], host="0.0.0.0")  # 调用run方法，设定端口号，启动服务

