#!/bin/bash

# The vm does not have enough drive space to store the docker images
# We will create a ramdisk for docker to use

sudo systemctl stop docker
sudo mkdir -p /tmp/ramdisk
sudo sudo mount -t tmpfs -o size=128G myramdisk /tmp/ramdisk
sudo rsync -avh /var/lib/docker /tmp/ramdisk/

echo " adding the following lines to /etc/docker/daemon.json"
echo " {"
echo "   \"data-root\": \"/tmp/ramdisk/docker\""
echo " }"
echo "Press enter to continue"
read

sudo nano /etc/docker/daemon.json

# request input from user
echo "Press enter to continue"
read

sudo systemctl start docker
cd ~/cloudfiles/code/Users/Joseffus.Santos/nachet-model-ccds/enviroments/pytorch-gpu-24
docker compose -f docker-compose.yml up -d
docker exec -it pytorch-gpu-24 /bin/bash