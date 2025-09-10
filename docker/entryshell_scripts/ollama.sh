#!/bin/sh

set -eu

# Start Ollama in the background
/bin/ollama serve &
pid=$!

# Give the daemon a moment to boot
sleep 5

# Wait for the daemon to respond (up to ~60s)
attempts=0
until /bin/ollama list >/dev/null 2>&1 || [ $attempts -ge 30 ]; do
  attempts=$((attempts + 1))
  sleep 2
done

# Determine model name
# Prefer MODEL, fallback to LOCAL_LLM_MODEL
RAW_MODEL_NAME="${MODEL:-${LOCAL_LLM_MODEL:-}}"

# Strip surrounding quotes if present
MODEL_NAME=$(echo "$RAW_MODEL_NAME" | tr -d '"')

# Normalize names like 'ollama/llama3.2:3b' -> 'llama3.2:3b'
case "$MODEL_NAME" in
  */*) MODEL_NAME=${MODEL_NAME#*/} ;;
esac

if [ -z "$MODEL_NAME" ]; then
  echo "❌ No model specified in MODEL/LOCAL_LLM_MODEL environment variable"
else
  if /bin/ollama list | grep -q "^$MODEL_NAME\\b" || /bin/ollama list | grep -q "$MODEL_NAME"; then
    echo "🟢 Model ($MODEL_NAME) already installed"
    : > /tmp/ollama_ready
  else
    echo "🔴 Retrieving model ($MODEL_NAME)..."
    if /bin/ollama pull "$MODEL_NAME" 2>/dev/null && /bin/ollama list | grep -q "$MODEL_NAME"; then
      echo "🟢 Model download complete!"
      : > /tmp/ollama_ready
    else
      echo "❌ Error downloading model ($MODEL_NAME)"
    fi
  fi
fi

# Wait for Ollama process to finish
wait "$pid"


