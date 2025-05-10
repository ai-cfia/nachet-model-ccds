# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
FROM nvcr.io/nvidia/pytorch:25.04-py3

# Define build arguments for username and password
ARG USERNAME=nachetuser
# Password will be passed as a build secret

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /nachetenv

# Install required packages
RUN apt update && apt install -y screen sudo openssh-server

# Configure SSH, generate SSH host keys, and create user with BuildKit secret
RUN --mount=type=secret,id=user_password \
    mkdir -p /var/run/sshd && \
    echo 'PermitRootLogin no' >> /etc/ssh/sshd_config && \
    echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config && \
    echo 'UsePrivilegeSeparation no' >> /etc/ssh/sshd_config && \
    # Generate SSH host keys
    ssh-keygen -A && \
    # Create user with sudo privileges
    useradd -m -s /bin/bash ${USERNAME} && \
    echo "${USERNAME}:$(cat /run/secrets/user_password 2>/dev/null || cat /dev/null)" | chpasswd && \
    usermod -aG sudo ${USERNAME} && \
    # Allow user to use sudo without password
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} && \
    chmod 0440 /etc/sudoers.d/${USERNAME}

# RUN curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' -o vscode_cli.tar.gz && tar -xf vscode_cli.tar.gz

COPY requirements-frozen.txt .

RUN pip install --no-cache-dir -r requirements-frozen.txt

# Expose SSH port
EXPOSE 22

# Change entrypoint to start SSH server and keep container running
CMD ["/usr/sbin/sshd", "-D"]
