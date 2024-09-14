FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /opt/CosyVoice

RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list \
  && apt-get update -y \
  && apt-get -y install git git-lfs

COPY requirements.txt .
# here we use python==3.10 because we cannot find an image which have both python3.8 and torch2.0.1-cu118 installed
RUN pip3 install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

COPY . .

EXPOSE 50000

CMD [ "python3", "./runtime/python/fastapi/server.py", "--model_dir", "./pretrained_models/CosyVoice-300M-Instruct" ]
