└─$ cat Dockerfile
# 基础镜像
FROM continuumio/miniconda3

COPY . /face/
RUN pip install -r /face/requirements.txt  -i  https://pypi.tuna.tsinghua.edu.cn/simple
WORKDIR /face

# 设置容器启动时的命令
#CMD ["python", "flask_http_server.py"]
CMD ["uvicorn", "fastapi_http_server:app", "--port", "30025" ,"--host" ,"0.0.0.0"]