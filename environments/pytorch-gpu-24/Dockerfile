# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
FROM nvcr.io/nvidia/pytorch:25.01-py3
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /nachetenv

RUN apt update
RUN apt install -y screen

COPY --from=project pyproject.toml .

RUN uv venv
RUN source .venv/bin/activate
RUN uv pip install -r pyproject.toml

ENTRYPOINT ["tail", "-f", "/dev/null"]
