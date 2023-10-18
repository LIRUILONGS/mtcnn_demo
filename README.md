# mtcnn_demo

mtcnn Demo

# HTTP 版本



## flask 版本
---

打了  Docker 镜像


默认使用 开发模式启动，提供了 gunicorn 部署环境，可以通过 docker 启动命令替换

```bash
(mtcnn) bash-4.2# docker run --rm  -p 30025:30025 mtcnn-hopenet-laplacian-face
```

gunicorn 方式

```bash
(mtcnn) bash-4.2# docker run --rm  -p 30025:30025 mtcnn-hopenet-laplacian-face  gunicorn -w 3  --worker-class gevent  -b 0.0.0.0:30025  --timeout 300  flask_http_server:app
```

gunicorn 方式并发高了会有一个报错，暂时没有解决

```bash
2023-10-12 03:17:42,310 - app.py[line:1414] - ERROR: Exception on /uploads [POST]
Traceback (most recent call last):
  File "/opt/conda/lib/python3.9/site-packages/flask/app.py", line 2190, in wsgi_app
    response = self.full_dispatch_request()
  File "/opt/conda/lib/python3.9/site-packages/flask/app.py", line 1486, in full_dispatch_request
    rv = self.handle_user_exception(e)
  File "/opt/conda/lib/python3.9/site-packages/flask/app.py", line 1484, in full_dispatch_request
    rv = self.dispatch_request()
  File "/opt/conda/lib/python3.9/site-packages/flask/app.py", line 1469, in dispatch_request
    return self.ensure_sync(self.view_functions[rule.endpoint])(**view_args)
  File "/face/flask_http_server.py", line 94, in decorated
    return f(*args, **kwargs)
  File "/face/flask_http_server.py", line 205, in uploads
    json_data = detect_face(body, filename)
  File "/face/flask_http_server.py", line 225, in detect_face
    faces =  context.mtcnn.detect_face(body, filename)
  File "/face/mtcnn_demo.py", line 117, in detect_face
    detections = self.face_detector.detect_faces(img_rgb)
  File "/opt/conda/lib/python3.9/site-packages/mtcnn/mtcnn.py", line 300, in detect_faces
    result = stage(img, result[0], result[1])
  File "/opt/conda/lib/python3.9/site-packages/mtcnn/mtcnn.py", line 342, in __stage1
    out = self._pnet.predict(img_y)
  File "/opt/conda/lib/python3.9/site-packages/keras/src/utils/traceback_utils.py", line 70, in error_handler
    raise e.with_traceback(filtered_tb) from None
  File "/opt/conda/lib/python3.9/site-packages/keras/src/utils/version_utils.py", line 126, in disallow_legacy_graph
    raise ValueError(error_msg)
ValueError: Calling `Model.predict` in graph mode is not supported when the `Model` instance was constructed with eager mode enabled. Please construct your `Model` instance in graph mode or call `Model.predict` with eager mode enabled.
```

镜像位置：

[https://hub.docker.com/repository/docker/liruilong/mtcnn-hopenet-laplacian-face/general](https://hub.docker.com/repository/docker/liruilong/mtcnn-hopenet-laplacian-face/general)

```bash
docker pull liruilong/mtcnn-hopenet-laplacian-face
```

## fastapi 版本

```bash
python  fastapi_http_server.py
```

## tomado 版本

```bash
python tornado_http_server.py
```
