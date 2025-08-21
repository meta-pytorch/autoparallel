#!/bin/bash
export PS4=' + [$(date +"%Y-%m-%d %H:%M:%S,%3N")] '
set -eExu -o pipefail

# want REGION_DATACENTER_PREFIX
source /etc/fbwhoami

#######################################
# Set up oilfs and airstore mounts if set.
# Globals:
#   DISABLE_OILFS kill switch to skip setting up oilfs and airstore entirely (default unset)
#   AI_RM_ATTRIBUTION should be set by mast for attribution
#######################################
function setup_oilfs {
  if [[ -n "${DISABLE_OILFS-}" ]]; then
    echo "OilFS disabled through env DISABLE_OILFS=$DISABLE_OILFS. Skipping mounts."
    return 0
  fi

  if [ -n "$ENABLE_AIRSTORE" ]; then
    FUSE_SRC="ws://ws.ai.pci0ai/genai_fair_llm"
  else
    FUSE_SRC="ws://ws.ai.nha0genai/checkpoint/infra"
  fi
  FUSE_DST="/mnt/wsfuse"

  mkdir -p "$FUSE_DST"
  /packages/oil.oilfs/oilfs-wrapper --profile="${OILFS_PROFILE-genai}" --user="$AI_RM_ATTRIBUTION" --log-level=debug "$FUSE_SRC" "$FUSE_DST"
}


#######################################
# Set up ManifoldFS mount if configured.
# Globals:
#   DISABLE_MANIFOLDFS kill switch to skip setting up manifoldfs (default unset)
#   MANIFOLDFS_FUSE_DST path on host to mount to; defaults to /mnt/mffuse
#   MANIFOLDFS_BUCKET which Manifold bucket to mount; if unset will skip setup
#######################################
function setup_manifoldfs {
  if [[ -n "${DISABLE_MANIFOLDFS-}" ]]; then
    echo "ManifoldFS disabled through env DISABLE_MANIFOLDFS=$DISABLE_MANIFOLDFS. Skipping mounts."
    return 0
  fi

  if [[ -z "${MANIFOLDFS_BUCKET-}" ]]; then
    echo "Manifold bucket is not set (MANIFOLDFS_BUCKET is empty), skipping setting up ManifoldFS"
    return 0
  fi

  MANIFOLDFS_FUSE_DST="${MANIFOLDFS_FUSE_DST:-/mnt/mffuse}"
  mkdir -p "${MANIFOLDFS_FUSE_DST}"

  if [[ -n "${ENABLE_MANIFUSE_OVER_MANIFOLDFS-}" ]]; then
    MANIFOLD_FUSE_SRC="manifold://$MANIFOLDFS_BUCKET/tree"
    /packages/oil.oilfs/oilfs-wrapper --profile="manifold" --user="$AI_RM_ATTRIBUTION" --log-level=debug "${MANIFOLD_FUSE_SRC}" "${MANIFOLDFS_FUSE_DST}"
  else
    MANIFOLDFS_BINARY=${MANIFOLDFS_BINARY:-"/packages/manifold.manifoldfs/manifoldfs"}
    "${MANIFOLDFS_BINARY}" "manifold.blobstore" "${MANIFOLDFS_BUCKET}" "${MANIFOLDFS_FUSE_DST}"
  fi
}

#######################################
# Mounts airstore with the right setup.
# Globals:
#   ENABLE_AIRSTORE enable airstore (default unset)
#   AIRSTORE_URI allows overriding the oilfs region used for airstore mount.
#######################################
function mount_airstore {
  if [[ -z "${ENABLE_AIRSTORE-}" ]]; then
    echo "Airstore has not been enabled through env ENABLE_AIRSTORE. Skipping mounts."
    return 0
  fi

  local airstore_uri="${AIRSTORE_URI-}"
  if [[ -z "$airstore_uri" ]]; then
    local host
    host="$(hostname)"

    case $host in
      *.pci* )
        airstore_uri="ws://ws.ai.pci0ai/airstore"
        ;;
      *.eag* )
        airstore_uri="ws://ws.ai.eag0genai/airstore"
        ;;
      *.gtn* )
        airstore_uri="ws://ws.ai.gtn0genai/airstore"
        ;;
      *.nha* )
        airstore_uri="ws://ws.ai.nha0genai/airstore"
        ;;
      *.snb* )
        airstore_uri="ws://ws.ai.snb0genai/airstore"
        ;;
      *.vcn* )
        airstore_uri="ws://ws.ai.vcn0genai/airstore"
        ;;
      *.zas* )
        airstore_uri="ws://ws.ai.zas0genai/airstore"
        ;;
      *.nao* )
        airstore_uri="ws://ws.ai.nao0ai/airstore"
        ;;
      * )
        echo -e "\e[31mNo airstore source available based on region of $host, only available in pci, eag, gtn, nha. You can mount a cross-region airstore by passing in the AIRSTORE_URI environment variable\e[0m" 1>&2
        exit 1
        ;;
    esac
  fi

  local mount_dir="${AIRSTORE_LOCAL_MOUNT_ROOT:-/data/users/airstore}"
  if [ ! -d "$mount_dir" ] ; then
    mkdir -p "$mount_dir"
  fi

  # Enable privacy logging for airstore mount unless pretraining
  if [[ "${OILFS_PROFILE-genai}" != "pretraining" ]]; then
    export OILFS_ENABLE_PRIVACY_LIB_LOGGER_AIRSTORE=1;
  fi

  echo "WS-Airstore: mount from $airstore_uri to $mount_dir"
  if [[ ${OILFS_USE_LEGACY_SCRIPT+set} && "${OILFS_USE_LEGACY_SCRIPT}" == 1  ]]; then
    /packages/oil.oilfs/scripts/airstore_wrapper.sh "$airstore_uri" "$mount_dir"
  else
    /packages/oil.oilfs/oilfs-wrapper --log-level debug --profile=airstore "$airstore_uri" "$mount_dir" --user "airstore-${AI_RM_ATTRIBUTION-}"
  fi
}

setup_oilfs
setup_manifoldfs
mount_airstore
