# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
FROM nvcr.io/nvidia/pytorch:24.12-py3
WORKDIR /nachetenv

RUN apt update
RUN apt install -y screen

RUN /usr/bin/python3 -m pip install --upgrade pip
COPY ["../../requirements.txt", "/nachetenv/"]

RUN pip install -r requirements.txt
