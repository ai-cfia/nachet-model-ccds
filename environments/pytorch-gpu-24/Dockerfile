# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
FROM nvcr.io/nvidia/pytorch:24.12-py3
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /nachetenv

RUN apt update && apt install -y screen
RUN curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' -o  vscode_cli.tar.gz && tar -xf vscode_cli.tar.gz

COPY requirements-frozen.txt .

RUN pip install --no-cache-dir -r requirements-frozen.txt

ENTRYPOINT ["tail", "-f", "/dev/null"]
