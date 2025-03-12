FROM ghcr.io/ai-cfia/nachet-model-ccds/serve-gpu-cu121-py311-prod:20250214

ENV PATH="/home/venv/bin:$PATH"

USER root

COPY --chown=model-server:model-server --chmod=755 artifacts/requirements-frozen.txt artifacts/config.properties artifacts/entrypoint.sh /home/model-server/

RUN pip install --no-cache-dir -r /home/model-server/requirements-frozen.txt

COPY --chown=model-server:model-server artifacts/*.mar /home/model-server/model-store/

# RUN apt update && apt install -y curl

USER model-server
ENTRYPOINT [ "/home/model-server/entrypoint.sh" ]
CMD [ "serve" ]
