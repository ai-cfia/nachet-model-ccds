#!/bin/bash
#!/bin/bash
set -e

echo "Starting TorchServe"
if [[ "$1" = "serve" ]]; then
    shift 1
    torchserve --start --ts-config /home/model-server/config.properties --disable-token-auth
else
    eval "$@"
fi

sleep 45
echo "Registering workflow"
curl -X POST "http://0.0.0.0:8081/workflows?url=file:///home/model-server/ensemble_27spp.war"

# prevent docker exit
tail -f /dev/null
