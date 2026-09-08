#!/bin/bash

set -euo pipefail
source /etc/fbwhoami

if [[ -z "${FUSE_SRC-}" ]]; then
    host=$(hostname)
    suffix=${FUSE_SRC_PATH:-checkpoint/infra}
    case ${host} in
        *.pci*) FUSE_SRC="ws://ws.ai.pci0ai/${suffix}" ;;
        *.eag*) FUSE_SRC="ws://ws.ai.eag0genai/${suffix}" ;;
        *.gtn*) FUSE_SRC="ws://ws.ai.gtn0genai/${suffix}" ;;
        *.nha*) FUSE_SRC="ws://ws.ai.nha0genai/${suffix}" ;;
        *.nao*) FUSE_SRC="ws://ws.ai.nao0genai/${suffix}" ;;
        *.snb*) FUSE_SRC="ws://ws.ai.snb0genai/${suffix}" ;;
        *.lco*) FUSE_SRC="ws://ws.ai.lco0genai/${suffix}" ;;
        *)
            echo "No OilFS source is configured for ${host}" >&2
            exit 1
            ;;
    esac
fi

FUSE_DST=/mnt/wsfuse
mkdir -p "${FUSE_DST}"
/packages/oil.oilfs/oilfs-wrapper \
    --profile="${OILFS_PROFILE:-genai}" \
    --user="${AI_RM_ATTRIBUTION}" \
    --log-level=debug \
    "${FUSE_SRC}" \
    "${FUSE_DST}"
