FROM ghcr.io/ai-cfia/nachet-model-ccds/serve-gpu-cu121-py311-prod:20250214

# Copy the model file
COPY artifacts/*.mar /home/model-server/model-store/
COPY artifacts/*.war /home/model-server/
COPY artifacts/config.properties /home/model-server/
COPY artifacts/requirements.txt /home/model-server/artifacts/
COPY artifacts/entrypoint.sh /home/model-server/

USER root

ENV PATH="/home/venv/bin:$PATH"
RUN chown -R model-server:model-server /home/model-server/
RUN chmod +x /home/model-server/entrypoint.sh
RUN apt update && apt install -y curl
RUN pip install --no-cache-dir -r /home/model-server/artifacts/requirements.txt

USER model-server
ENTRYPOINT [ "/home/model-server/entrypoint.sh" ]
CMD [ "serve" ]
