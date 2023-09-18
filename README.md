# mtcnn_demo

mtcnn Demo


这是一个 人脸检测的 Demo， 用于输出适合人脸识别的 人脸数据集，通过 mtcnn 检测人脸，拿到置信度，通过 Hopenet 确定人脸是姿态，拿到姿态欧拉角，通过 拉普拉斯算子 确定人脸模糊度。

这里默认对人脸做了对齐处理，通过 opencv 的 透视变化方法实现

# HTTP 版本
---

## flasl 版本




## tornado 版本

使用  tornado 构建 web 服务，目前只支持 token 认证
 
```bash
(mtcnn) C:\Users\liruilong\Documents\GitHub\mtcnn_demo>python tornado_http_server.py
2023-09-15 01:35:39,939 - tornado_http_server.py[line:165] - INFO: 🚀 服务启动中
2023-09-15 01:35:39,940 - tornado_http_server.py[line:140] - INFO: 🚀🚀 路由表信息加载
2023-09-15 01:35:39,946 - tornado_http_server.py[line:156] - INFO: 🚀🚀🚀 人脸检测相关模型加载
2023-09-15 01:35:40,860 - tornado_http_server.py[line:158] - INFO: 🚀🚀🚀🚀 构建上下文对象
2023-09-15 01:35:40,861 - tornado_http_server.py[line:161] - INFO: 🚀🚀🚀🚀🚀 服务启动成功
1/1 [==============================] - 0s 312ms/step
1/1 [==============================] - 0s 168ms/step
1/1 [==============================] - 0s 63ms/step
1/1 [==============================] - 0s 44ms/step
1/1 [==============================] - 0s 34ms/step
1/1 [==============================] - 0s 21ms/step
1/1 [==============================] - 0s 22ms/step
1/1 [==============================] - 0s 20ms/step
1/1 [==============================] - 0s 18ms/step
1/1 [==============================] - 0s 22ms/step
1/1 [==============================] - 0s 21ms/step
1/1 [==============================] - 0s 21ms/step
35/35 [==============================] - 0s 5ms/step
2/2 [==============================] - 0s 9ms/step
⚠️: 19e2304a0fae4c17be7b69f1fcb13513 中该置信度 0.939575731754303  未达到阈值 0.995，被弃用
2023-09-15 01:35:44,095 - web.py[line:2344] - INFO: 200 POST /upload (127.0.0.1) 3233.51ms
```



接口信息见： `New Collection.postman_collection.json`


+ `/` ： 欢迎页
+ `/livez` ： 存活探针
+ `/readyz`： 就绪探针
+ `/upload`： 上传文件解析
+ `/uploads`： 上传多文件解析

解析接口需要传 token

## 解析接口返回信息


```json
{   
    "image_id": "19e2304a0fae4c17be7b69f1fcb13513",
    "face_total": 4,
    "face_efficient_total_resp": 3,
    "resp": [
        {
            "face_id": "af1a07cb20c04adcbf27d6315da8f0e2",
            "face_blur": 513.2548600291768,
            "face_pose": {
                "pitch": -14.191238403320312,
                "yaw": -22.01685333251953,
                "roll": -1.958282470703125
            },
            "face_confidence": 0.9999302625656128,
            "face_coordinate": [
                571,
                662,
                59,
                85
            ],
            "facie5points": {
                "left_eye": [
                    579,
                    696
                ],
                "right_eye": [
                    605,
                    700
                ],
                "nose": [
                    588,
                    716
                ],
                "mouth_left": [
                    580,
                    728
                ],
                "mouth_right": [
                    604,
                    730
                ]
            },
            "face_native_image_b64": "/9j/4A.................AQSkZ",
            "face_native_images_b64": "iVBORw0........C",
            "face_align_images_b64": "iVBOR..........QmCC"
        },
        ,
       .........
    ],
    "mark_image_face_b64": "/9jn...................//2Q=="
}

```
## 调用方式

### curl 

```bash
curl --location --request POST 'http://127.0.0.1:30025/upload' \
--header 'Authorization: token' \
--form 'image=@"/C:/Users/liruilong/Pictures/vlcsnap-2023-06-18-22h34m23s680.png"'```
```

### js

```js
var myHeaders = new Headers();
myHeaders.append("Authorization", "token");

var formdata = new FormData();
formdata.append("image", fileInput.files[0], "/C:/Users/liruilong/Pictures/vlcsnap-2023-06-18-22h34m23s680.png");

var requestOptions = {
  method: 'POST',
  headers: myHeaders,
  body: formdata,
  redirect: 'follow'
};

fetch("http://127.0.0.1:30025/upload", requestOptions)
  .then(response => response.text())
  .then(result => console.log(result))
  .catch(error => console.log('error', error));
```
### python 

```py
import requests

url = "http://127.0.0.1:30025/upload"

payload={}
files=[
  ('image',('vlcsnap-2023-06-18-22h34m23s680.png',open('/C:/Users/liruilong/Pictures/vlcsnap-2023-06-18-22h34m23s680.png','rb'),'image/png'))
]
headers = {
  'Authorization': 'token'
}

response = requests.request("POST", url, headers=headers, data=payload, files=files)

print(response.text)

```

多文件请求报文

```bash
POST /uploads HTTP/1.1
Host: 127.0.0.1:30025
Authorization: token
Content-Length: 998
Content-Type: multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW

----WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="image"; filename="/C:/Users/liruilong/Pictures/vlcsnap-2023-08-10-02h42m20s009.png"
Content-Type: image/png

(data)
----WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="image"; filename="/C:/Users/liruilong/Pictures/vlcsnap-2023-08-10-02h42m20s358.png"
Content-Type: image/png

(data)
----WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="image"; filename="/C:/Users/liruilong/Pictures/vlcsnap-2023-08-10-02h42m20s687.png"
Content-Type: image/png

(data)
----WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="image"; filename="/C:/Users/liruilong/Pictures/vlcsnap-2023-08-10-02h42m20s996.png"
Content-Type: image/png

(data)
----WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="image"; filename="/C:/Users/liruilong/Pictures/vlcsnap-2023-08-10-02h42m21s326.png"
Content-Type: image/png

(data)
----WebKitFormBoundary7MA4YWxkTrZu0gW

```

### tornado 编写异步代码


1. 基于生成器的协程
协程使用 Python 中的关键字 yield 来替代链式回调来实现挂起和继续程序的执行，需要使用gen.coroutine 来装饰生成器函数


```py
import tornado.ioloop
from tornado.web import RequestHandler, Application
from tornado.httpserver import HTTPServer
from tornado.options import options, define
from tornado.httpclient import AsyncHTTPClient
from tornado import gen

define('port', default=8000, help='监听端口')


class HelloHandler(RequestHandler):
    @gen.coroutine
    def get(self):
        url = "http://coolpython.net"
        response_code = yield self.fetch_coroutine(url)
        self.finish(str(response_code))

    @gen.coroutine
    def fetch_coroutine(self, url):
        http_client = AsyncHTTPClient()
        response = yield http_client.fetch(url)
        return response.code

if __name__ == '__main__':
    options.parse_command_line()
    handlers_routes = [
        (r'/', HelloHandler)
    ]
    app = Application(handlers=handlers_routes)
    http_server = HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.current().start()

```

从处理请求的get方法开始，就需要使用gen.coroutine 装饰器，fetch_coroutine 是一个协程，调用它时，必须加上yield 关键字，这样get方法也是一个基础生成器的协程。

```py
class HelloHandler(RequestHandler):

    async def get(self):
        url = "http://coolpython.net"
        response_code = await self.fetch_coroutine(url)
        self.finish(str(response_code))

    async def fetch_coroutine(self, url):
        http_client = AsyncHTTPClient()
        response = await http_client.fetch(url)
        return response.code
```

CPU计算过于密集，那么此种情况下tornado无能为力，单进程条件下，CPU导致的阻塞是没有办法进行异步处理的。tornado 能够异步处理的是那些网络IO操作，所以并不合适