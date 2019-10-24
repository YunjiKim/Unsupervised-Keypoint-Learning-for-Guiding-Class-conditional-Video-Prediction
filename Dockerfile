
FROM nvidia/cuda:9.0-cudnn7-devel

RUN apt-get update && apt-get install python3 python3-pip -y
RUN ln -s /usr/bin/python3 /usr/bin/python && ln -s /usr/bin/pip3 /usr/bin/pip

ADD requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

WORKDIR /workspace