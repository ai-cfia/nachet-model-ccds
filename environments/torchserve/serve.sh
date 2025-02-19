#!/bin/bash
torchserve --start --model-store /home/model-server/model-store --models 27spp_model_1=27spp_model_1.mar --enable-model-api --disable-token-auth --ts-config /home/model-server/config.properties 

