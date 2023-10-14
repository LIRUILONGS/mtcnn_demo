#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   fastapi_http_server.py
@Time    :   2023/10/13 20:56:22
@Author  :   Li Ruilong
@Version :   1.0
@Contact :   liruilonger@gmail.com
@Desc    :   fastapi 版本的 http 服务
"""


# here put the import lib
"""
pip install fastapi
"""


from globals import GlobalObject
import yaml_utils as Yaml
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Union
from PIL import Image
import utils
import logging
import json
import os
from fastapi import BackgroundTasks, FastAPI, UploadFile, File, HTTPException, Depends
app = FastAPI()
security = HTTPBearer()
# 实例化并命名为app实例
logging.basicConfig(level=logging.INFO, )


global_object = GlobalObject()

config = Yaml.get_yaml_config(file_name="config/config.yaml")
fastapi_config = config['fastapi']




async def get_current_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    
    """
    @Time    :   2023/10/13 21:28:08
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   token 认证
    """
    
    token = credentials.credentials

    # 在这里实现你的 Token 认证逻辑，例如验证 Token 的有效性、解码 Token 获取用户信息等
    # 如果认证失败，你可以抛出一个 HTTPException，返回 401 Unauthorized 错误
    valid_tokens = fastapi_config["token"]
    if not token:
        raise HTTPException(status_code=401, detail="Token is missing")
    logging.info(token)
    logging.info(valid_tokens)
    if token !=  valid_tokens:
        raise HTTPException(status_code=401, detail="Invalid token")
    return True


@app.get("/")
def index():
    """
    @Time    :   2023/09/17 22:40:15
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   欢迎页
    """

    return {'result': "Hello, metcnn face"}


@app.get("/livez")
def livez():
    """
    @Time    :   2023/09/17 22:44:43
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   服务存活探针接口
    """
    return {'result': "mtcnn live  ^_^"}


@app.get("/token")
def token():
    return {
        "code": 200,
        "message": "获取 token",
        "token": fastapi_config["token"],
    }


@app.get("/readyz")
def readyz():
    """
    @Time    :   2023/09/17 22:46:20
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   服务就绪探针接口
    """
    if global_object.ready_mark:
        return {'result': "mtcnn ready ^_^ "}
    else:
        raise HTTPException(
            status_code=503,
            detail="Item not found",
            headers={"X-Error": "Service Unavailable"},
        )


@app.post("/upload")
async def upload(image: UploadFile,token: str = Depends(get_current_token)):
    """
    @Time    :   2023/09/18 01:41:00
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   单文件上传检测
    """
    try:
        file = image
        if file.filename == '':
            raise HTTPException(status_code=404, detail="Upload failed, no files found ^_^")
        filename = file.filename
        body = await file.read()
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Upload failed, no files found ^_^{e}")
    try:
        # 验证图片完整性
        if utils.is_image_file(body):
            json_data = await detect_face(body, filename)
            return json.loads(json_data)
        else:
            raise HTTPException(status_code=400, detail=f"File {filename} uploaded successfully, parsing failed, image incomplete ^_^")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File {filename} uploaded successfully, parsing failed for unknown reason ^_^{e}")



async def detect_face(body, filename):
    """
    @Time    :   2023/09/17 21:27:47
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   
    """
    faces = global_object.mtcnn.detect_face(body, filename)
    json_data = json.dumps(faces)
    return json_data


