#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   tornado_http_server.py
@Time    :   2023/10/10 01:39:12
@Author  :   Li Ruilong
@Version :   1.0
@Contact :   liruilonger@gmail.com
@Desc    :   tornado 版本 HTTPd 服务
"""

# here put the import lib

import asyncio

import tornado.web
import logging

from mtcnn_demo import MtcnnDetectFace
import utils
import json
import yaml_utils as Yaml
import time
from tornado import gen
from tornado.options import options, define



tornado.log.enable_pretty_logging()



face_log = logging.basicConfig(level=logging.INFO, )

config=  Yaml.get_yaml_config(file_name="config/config.yaml")
tornado_config = config['tornado']


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


def token_auth_middleware(handler_class):
    """
    @Time    :   2023/09/14 21:51:23
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   令牌认证装饰器
    """
    class TokenAuthMiddleware(handler_class):
        # tocker 通过配置文件加载
        valid_tokens = tornado_config["token"]

        def prepare(self):
            token = self.request.headers.get("Authorization")
            if not token:
                self.set_status(401)
                self.finish("Unauthorized")
                return

            if token not in self.valid_tokens:
                self.set_status(403)
                self.finish("Forbidden")
                return

    return TokenAuthMiddleware

class TokenHandler(tornado.web.RequestHandler):
    """
    @Time    :   2023/09/14 05:52:27
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   获取 token
    """

    async def get(self):
        valid_tokens = tornado_config["token"]
        self.write(valid_tokens)


class MainHandler(tornado.web.RequestHandler):
    """
    @Time    :   2023/09/14 05:52:27
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   欢迎页
    """

    async def get(self):
        self.write("Hello, face")


class ReadinessProbeHandler(tornado.web.RequestHandler):
    """
    @Time    :   2023/09/14 05:52:47
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   服务就绪探针接口
    """

    async  def get(self):
        context = self.application.my_context
        if context.ready_mark:
            self.write(" ready  ^_^")
        else:
            self.set_status(503)
            self.finish("Service Unavailable")
            return


class LivenessProbeHandler(tornado.web.RequestHandler):
    """
    @Time    :   2023/09/14 05:53:31
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   服务存活探针接口
    """

    async  def get(self):
        self.write(str(time.time()) +  " live ^_^ ")


@token_auth_middleware
class UploadHandler(tornado.web.RequestHandler):
    """
    @Time    :   2023/09/14 22:35:57
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   人脸文件上传检测
    """
    @gen.coroutine
    def post(self):
        """
        @Time    :   2023/09/14 23:03:32
        @Author  :   liruilonger@gmail.com
        @Version :   1.0
        @Desc    :   通过HTTP 上传文件解析
        """
        try:
            file = self.request.files.get('image')[0]
            filename = file['filename']
            body = file['body']
        except:
            self.set_status(400)
            self.finish("Upload failed, no files found ^_^")
            return
        try:
            # 验证图片完整性
            if utils.is_image_file(body):
                json_data =  yield self.detect_face(body, filename)
                self.write(json_data)
            else:
                self.set_status(400)
                self.finish(f"File {filename} uploaded successfully, parsing failed, image incomplete ^_^")                
        except:
            self.set_status(500)
            self.finish(
                f"File {filename} uploaded successfully, parsing failed for unknown reason ^_^")
    
    @gen.coroutine
    def detect_face(self,body, filename):
        """
        @Time    :   2023/09/17 21:27:47
        @Author  :   liruilonger@gmail.com
        @Version :   1.0
        @Desc    :   基于生成器的协程
        """
        context = self.application.my_context
        faces =  context.mtcnn.detect_face(body, filename)
        json_data = json.dumps(faces)
        return json_data



@token_auth_middleware
class UploadsHandler(tornado.web.RequestHandler):
    """
    @Time    :   2023/09/14 22:35:57
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   人脸文件上传检测:多张照片
    """

    async def post(self):
        """
        @Time    :   2023/09/14 23:03:32
        @Author  :   liruilonger@gmail.com
        @Version :   1.0
        @Desc    :   通过HTTP 上传文件解析
        """
        try:
            files = self.request.files.get('image')
            face_file = []
            for file in files:
                face_file.append({
                    "filename": file['filename'],
                    "body": file['body']
                })
        except:
            self.set_status(400)
            self.finish("Upload failed, no files found ^_^")
            return
        try:
            face_imges = []
            for file in face_file:
                body = file["body"]
                filename = file["filename"]
                # 验证图片完整性

                if utils.is_image_file(body):
                    faces = await   self.detect_face(body, filename)
                    face_imges.append(faces)

            json_data = json.dumps(face_imges)
            self.set_header('Content-Type', 'application/json')
            self.write(json_data)
                    
        except:
            self.write(f"文件  {filename}  上传成功，解析失败 ^_^ ")

    async def detect_face(self,body, filename):
        """
        @Time    :   2023/09/17 21:27:47
        @Author  :   liruilonger@gmail.com
        @Version :   1.0
        @Desc    :    async 和 await 关键字
        """
        context = self.application.my_context
        faces =  context.mtcnn.detect_face(body, filename)
        json_data = json.dumps(faces)
        return json_data

def make_app():
    """
    @Time    :   2023/09/14 22:08:51
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   路由表信息加载
    """
    logging.info('🚀 路由表信息加载')

    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/token", TokenHandler),
        (r"/livez", LivenessProbeHandler),
        (r"/readyz", ReadinessProbeHandler),
        (r"/upload", UploadHandler),
        (r"/uploads", UploadsHandler)
    ],
    )


async def main():
    app = make_app()
    app.listen(tornado_config["port"],address="")
    mtcnn = MtcnnDetectFace()
    # 加载模型
    logging.info('🚀🚀🚀🚀 人脸检测相关模型加载')
    mtcnn.build_model()
    logging.info('🚀🚀🚀🚀 构建上下文对象')
    my_context = MyContextObject(mtcnn, ready_mark=True)
    app.my_context = my_context
    logging.info('🚀🚀🚀🚀🚀 \033[32m服务启动成功\033[0m')
    await asyncio.Event().wait()
    

if __name__ == "__main__":
    logging.info('🚀 \033[32m服务启动中\033[0m')
    asyncio.run(main())
