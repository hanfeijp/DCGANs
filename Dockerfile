# GPU
# FROM nvidia/cuda:8.0-cudnn5-runtime

# CPU
FROM ubuntu:16.04

LABEL maintainer "example@example.jp"

# LABEL com.example.version="0.0.1"
# LABEL vendor="Housmart Inc"
# LABEL com.example.release-date="2017-00-00"
# LABEL com.example.version.is-production=""

RUN apt-get update
RUN apt-get -y install python3-pip curl
RUN pip3 install keras tensorflow-gpu jupyter opencv-python
RUN pip3 install --upgrade pip
RUN apt-get -y install python3.5 python3.5-dev
RUN apt-get -y install python3-numpy python3-scipy python3-matplotlib

ADD DCGAN.py data/DCGAN.py
ADD seen_batch.pickle data/seen_batch.pickle

ENTRYPOINT ["bash"]
#CMD ["python3 data/DCGANs.py", "-D", "FOREGROUND"]