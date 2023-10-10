# 基础镜像
FROM continuumio/miniconda3

# 将环境配置文件复制到镜像中
COPY . /

# 创建并激活环境
RUN conda env create -f /environment.yml
RUN echo "source activate mtcnn" > ~/.bashrc
RUN pip install -r /requirements.txt  -i  https://pypi.tuna.tsinghua.edu.cn/simple
ENV PATH /opt/conda/envs/mtcnn/bin:$PATH

# 复制你的应用程序代码到镜像中
WORKDIR /

# 设置容器启动时的命令
CMD ["python", "flask_http_server.py"]
