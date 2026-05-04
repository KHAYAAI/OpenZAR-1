#!/usr/bin/env bash
# Deploy the ZAR-1 Zenith canister to the Zenith testnet.
#
# Prereqs:
#   - rustup with `wasm32-unknown-unknown` target
#   - `zenith-cli` installed and authenticated (ZENITH_TOKEN set)
#   - configs/zar1_7b.yaml exists
#   - Trained checkpoint at $CHECKPOINT (defaults to checkpoints/zar1_7b/latest.pt)

set -euo pipefail

CONFIG=${CONFIG:-configs/zar1_7b.yaml}
CHECKPOINT=${CHECKPOINT:-checkpoints/zar1_7b/latest.pt}
NETWORK=${NETWORK:-testnet}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CRATE_DIR="$ROOT/zenith/canister"

echo "[1/5] Exporting ONNX model..."
python "$ROOT/scripts/export_onnx.py" \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output "$CRATE_DIR/zar1.onnx" \
    --seq-len 512 \
    --n-loops 4

echo "[2/5] Installing wasm32 target..."
rustup target add wasm32-unknown-unknown >/dev/null

echo "[3/5] Building canister WASM (release)..."
(
    cd "$CRATE_DIR"
    cargo build --release --target wasm32-unknown-unknown --features zk
)

WASM_PATH="$CRATE_DIR/target/wasm32-unknown-unknown/release/zar1_zenith_canister.wasm"
echo "[4/5] WASM artifact: $WASM_PATH ($(du -h "$WASM_PATH" | cut -f1))"

echo "[5/5] Deploying to Zenith ($NETWORK)..."
zenith-cli canister deploy \
    --network "$NETWORK" \
    --wasm "$WASM_PATH" \
    --name zar1 \
    --memory 4GB \
    --enable-zk \
    --batch-size 1000

echo "Deployed. Use 'zenith-cli canister info zar1 --network $NETWORK' to inspect."
