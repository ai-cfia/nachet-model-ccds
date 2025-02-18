FROM ghcr.io/ai-cfia/nachet-model-ccds/serve-gpu-cu121-py311-prod:20250214

# Copy the model file
COPY artifacts /home/model-server/

USER root
RUN chown -R model-server:model-server /home/model-server/
RUN mv /home/model-server/artifacts/27spp_model_1.mar /home/model-server/model-store/27spp_model_1.mar

ENV PATH="/home/venv/bin:$PATH"
RUN pip install --no-cache-dir -r /home/model-server/artifacts/requirements.txt

USER model-server
# ENTRYPOINT [ "torchserve", "--start", "--model-store", "/home/model-server/model-store", "--models", "27spp_model_1.mar", "--enable-model-api", "--ts-config", "/home/model-server/config.properties", "--disable-token-auth" ]
# ENTRYPOINT [ "torchserve", "--start", "--model-store", "/home/model-server/model-store", "--models", "27spp_model_1=27spp_model_1.mar", "--enable-model-api", "--disable-token-auth" ]
# torchserve --start --model-store /home/model-server/model-store --models 27spp_model_1=27spp_model_1.mar --enable-model-api --ts-config /home/model-server/config.properties --disable-token-auth
# ENTRYPOINT ["tail", "-f", "/dev/null"]
