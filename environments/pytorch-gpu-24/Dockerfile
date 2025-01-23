# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
FROM nvcr.io/nvidia/pytorch:24.12-py3
WORKDIR /nachetenv

RUN apt update
RUN apt install -y screen pipx && pipx ensurepath
RUN pipx install uv

COPY --from=project pyproject.toml .

RUN uv pip install -r pyproject.toml
