FROM nvidia/cuda:8.0-cudnn5-runtime

LABEL maintainer "example@example.jp"

RUN apt-get update
RUN apt-get -y install python3-pip curl
RUN pip3 install keras tensorflow-gpu jupyter opencv-python
RUN apt-get -y install python3.5 python3.5-dev
RUN apt-get -y install python3-numpy python3-scipy python3-matplotlib
