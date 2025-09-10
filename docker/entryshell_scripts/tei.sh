#!/bin/sh

set -eu

MODEL_ID="${LOCAL_EMBEDDINGS_MODEL:-intfloat/multilingual-e5-large-instruct}"
PORT="${TEI_PORT:-8504}"

exec text-embeddings-router --model-id "$MODEL_ID" --port "$PORT"


