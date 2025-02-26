# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
FROM nvcr.io/nvidia/pytorch:25.02-py3
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /nachetenv

RUN apt update
RUN apt install -y screen

COPY requirements-frozen.txt .

RUN pip install -r requirements-frozen.txt

ENTRYPOINT ["tail", "-f", "/dev/null"]
