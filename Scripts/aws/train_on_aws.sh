#!/usr/bin/env bash
set -euo pipefail

# --- Locate & load env ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/aws.env"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "❌ Missing $ENV_FILE. Copy aws.env.example -> aws.env and fill your values."
  exit 1
fi
# shellcheck disable=SC1090
source "$ENV_FILE"

export AWS_PAGER=""

say() { echo -e "[$(date +%H:%M:%S)] $*"; }
req() { local k="$1"; [[ -n "${!k:-}" ]] || { echo "❌ Missing env var: $k"; exit 1; }; }

# --- Required vars (from aws.env) ---
req AWS_PROFILE
req AWS_REGION
req INSTANCE_TYPE
req SECURITY_GROUP_NAME
req KEY_NAME
req SSH_USERNAME
req REPO_URL
req GIT_BRANCH
req REMOTE_TRAIN_SCRIPT
req REMOTE_DATASET
req REMOTE_SPLIT
req TRAIN_OUT_DIR
# Either AMI_ID or AMI_NAME_FILTER must be provided
if [[ -z "${AMI_ID:-}" && -z "${AMI_NAME_FILTER:-}" ]]; then
  echo "❌ Provide AMI_ID or AMI_NAME_FILTER in aws.env"
  exit 1
fi
VOLUME_SIZE_GB="${VOLUME_SIZE_GB:-100}"
USE_SPOT="${USE_SPOT:-false}"
MAX_PRICE_PER_HOUR="${MAX_PRICE_PER_HOUR:-1.00}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-20}"
TRAIN_BATCH="${TRAIN_BATCH:-16}"
TRAIN_LR="${TRAIN_LR:-3e-4}"
DEVICE="${DEVICE:-cuda}"

# Optional: local dataset folder to SCP (pairs.npz, split.json, vocab.json)
LOCAL_DATASET_DIR="${LOCAL_DATASET_DIR:-}"

# --- Resolve AMI if needed ---
if [[ -z "${AMI_ID:-}" ]]; then
  say "🔎 Resolving AMI by name filter: ${AMI_NAME_FILTER}"
  AMI_ID="$(aws --profile "$AWS_PROFILE" --region "$AWS_REGION" ec2 describe-images \
    --owners amazon \
    --filters "Name=name,Values=${AMI_NAME_FILTER}" "Name=state,Values=available" \
    --query 'Images | sort_by(@,&CreationDate)[-1].ImageId' --output text)"
  if [[ -z "$AMI_ID" || "$AMI_ID" == "None" ]]; then
    echo "❌ Failed to resolve AMI by filter: ${AMI_NAME_FILTER}"
    exit 1
  fi
fi
say "✅ Using AMI: ${AMI_ID}"

# --- Ensure key pair (downloads .pem if created) ---
PEM_PATH="${SCRIPT_DIR}/${KEY_NAME}-${AWS_REGION}.pem"
if ! aws --profile "$AWS_PROFILE" --region "$AWS_REGION" ec2 describe-key-pairs \
     --key-names "$KEY_NAME" >/dev/null 2>&1; then
  say "🔐 Creating key pair: ${KEY_NAME}"
  aws --profile "$AWS_PROFILE" --region "$AWS_REGION" ec2 create-key-pair \
    --key-name "$KEY_NAME" \
    --query "KeyMaterial" --output text > "$PEM_PATH"
  chmod 400 "$PEM_PATH"
else
  # If key already exists, ensure you have a .pem locally (script won't fetch old material)
  if [[ ! -f "$PEM_PATH" ]]; then
    say "⚠️ Key pair exists in AWS but ${PEM_PATH} not found locally."
    say "   Create/locate the matching .pem or set KEY_NAME to a new key."
    exit 1
  fi
fi

# --- Ensure security group ---
DEFAULT_VPC_ID="$(aws --profile "$AWS_PROFILE" --region "$AWS_REGION" ec2 describe-vpcs \
  --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)"
if [[ -z "$DEFAULT_VPC_ID" || "$DEFAULT_VPC_ID" == "None" ]]; then
  echo "❌ Could not find default VPC in ${AWS_REGION}"
  exit 1
fi

if ! SG_ID="$(aws --profile "$AWS_PROFILE" --region "$AWS_REGION" ec2 describe-security-groups \
        --group-names "$SECURITY_GROUP_NAME" --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null)"; then
  SG_ID=""
fi

if [[ -z "$SG_ID" || "$SG_ID" == "None" ]]; then
  say "🛡️  Creating security group: ${SECURITY_GROUP_NAME}"
  SG_ID="$(aws --profile "$AWS_PROFILE" --region "$AWS_REGION" ec2 create-security-group \
    --group-name "$SECURITY_GROUP_NAME" --description "SoloMuse training SG" \
    --vpc-id "$DEFAULT_VPC_ID" --query 'GroupId' --output text)"
  # Lock SSH to your current IP if possible
  MYIP="$(curl -s https://checkip.amazonaws.com || true)"
  CIDR="0.0.0.0/0"
  [[ -n "$MYIP" ]] && CIDR="${MYIP}/32"
  say "🔓 Authorizing inbound SSH 22 from ${CIDR}"
  aws --profile "$AWS_PROFILE" --region "$AWS_REGION" ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" --protocol tcp --port 22 --cidr "$CIDR" >/dev/null
else
  say "✅ Using security group ${SECURITY_GROUP_NAME} (${SG_ID})"
fi

# --- Block device (EBS size) ---
BDM_JSON="$(cat <<JSON
[
  {
    "DeviceName": "/dev/sda1",
    "Ebs": {
      "VolumeSize": ${VOLUME_SIZE_GB},
      "VolumeType": "gp3",
      "DeleteOnTermination": true
    }
  }
]
JSON
)"

# --- Launch instance ---
RUN_ARGS=(--profile "$AWS_PROFILE" --region "$AWS_REGION" ec2 run-instances
  --image-id "$AMI_ID"
  --instance-type "$INSTANCE_TYPE"
  --key-name "$KEY_NAME"
  --security-group-ids "$SG_ID"
  --block-device-mappings "$BDM_JSON"
  --count 1
)
if [[ "${USE_SPOT}" == "true" ]]; then
  say "🧾 Requesting SPOT instance (max \$${MAX_PRICE_PER_HOUR}/h)"
  RUN_ARGS+=(--instance-market-options "MarketType=spot")
fi

say "🚀 Launching ${INSTANCE_TYPE}…"
INSTANCE_ID="$( "${RUN_ARGS[@]}" --query 'Instances[0].InstanceId' --output text )"
say "✅ Instance ID: ${INSTANCE_ID}"

# Tag it for sanity
aws --profile "$AWS_PROFILE" --region "$AWS_REGION" ec2 create-tags \
  --resources "$INSTANCE_ID" --tags Key=Name,Value=SoloMuseTrainer >/dev/null

# Wait until running & OK
aws --profile "$AWS_PROFILE" --region "$AWS_REGION" ec2 wait instance-running --instance-ids "$INSTANCE_ID"
aws --profile "$AWS_PROFILE" --region "$AWS_REGION" ec2 wait instance-status-ok --instance-ids "$INSTANCE_ID"

PUBLIC_DNS="$(aws --profile "$AWS_PROFILE" --region "$AWS_REGION" ec2 describe-instances \
  --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicDnsName' --output text)"
say "🌐 Public DNS: ${PUBLIC_DNS}"

SSH="ssh -o StrictHostKeyChecking=no -i \"$PEM_PATH\" ${SSH_USERNAME}@${PUBLIC_DNS}"
SCP="scp -o StrictHostKeyChecking=no -i \"$PEM_PATH\""

# --- Remote setup & training ---
say "📦 Provisioning instance & starting training…"
$SSH bash -lc "
  set -e
  # DLAMI typically has conda & CUDA ready
  if command -v conda >/dev/null 2>&1; then
    source /etc/profile.d/conda.sh || true
    conda activate pytorch || true
  fi

  sudo apt-get update -y
  sudo apt-get install -y git git-lfs tmux
  git lfs install

  if [ ! -d SoloMuse-main ]; then
    git clone '$REPO_URL' SoloMuse-main
  fi
  cd SoloMuse-main
  git fetch origin '$GIT_BRANCH' || true
  git checkout '$GIT_BRANCH'

  python -m pip install --upgrade pip
  if [ -f requirements.txt ]; then
    python -m pip install -r requirements.txt
  fi

  # Ensure dataset directories exist
  mkdir -p \"\$(dirname \"$REMOTE_DATASET\")\"
  mkdir -p \"\$(dirname \"$REMOTE_SPLIT\")\"
  # vocab is optional in your current trainer, but we support it if present
"

# If you have local dataset files, upload them now
if [[ -n "$LOCAL_DATASET_DIR" && -d "$LOCAL_DATASET_DIR" ]]; then
  say "⤴️  Uploading dataset from local: ${LOCAL_DATASET_DIR}"
  $SCP "${LOCAL_DATASET_DIR}/pairs.npz"  "${SSH_USERNAME}@${PUBLIC_DNS}:/home/${SSH_USERNAME}/pairs.npz"
  $SCP "${LOCAL_DATASET_DIR}/split.json" "${SSH_USERNAME}@${PUBLIC_DNS}:/home/${SSH_USERNAME}/split.json" || true
  $SCP "${LOCAL_DATASET_DIR}/vocab.json" "${SSH_USERNAME}@${PUBLIC_DNS}:/home/${SSH_USERNAME}/vocab.json" || true

  $SSH bash -lc "
    set -e
    cd SoloMuse-main
    mv ~/pairs.npz  \"$REMOTE_DATASET\"
    [ -f ~/split.json ] && mv ~/split.json \"$REMOTE_SPLIT\" || true
    [ -f ~/vocab.json ] && cp ~/vocab.json Training/datasets/slakh_micro/vocab.json || true
  "
fi

# Kick off training inside tmux so it survives disconnection
say "🧠 Starting train job in tmux (session: train)…"
$SSH bash -lc "
  set -e
  cd SoloMuse-main
  tmux new -d -s train \
   \"python '$REMOTE_TRAIN_SCRIPT' \
      --data '$REMOTE_DATASET' \
      --split '$REMOTE_SPLIT' \
      --out '$TRAIN_OUT_DIR' \
      --epochs $TRAIN_EPOCHS \
      --batch_size $TRAIN_BATCH \
      --lr $TRAIN_LR\"
  tmux ls
  echo 'Training started. To attach: tmux attach -t train'
"
say "✅ Launched. Attach with:  ssh -i \"$PEM_PATH\" ${SSH_USERNAME}@${PUBLIC_DNS} && tmux attach -t train"

echo
say "🧹 When finished, stop or terminate to save $:"
echo "  aws --profile \"$AWS_PROFILE\" --region \"$AWS_REGION\" ec2 stop-instances --instance-ids \"$INSTANCE_ID\""
echo "  aws --profile \"$AWS_PROFILE\" --region \"$AWS_REGION\" ec2 terminate-instances --instance-ids \"$INSTANCE_ID\""