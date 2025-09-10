#!/bin/sh

set -eu

# Render LiteLLM config from template
python /app/render_litellm_config.py /app/utility/litellm.config.template.yaml /app/utility/litellm.config.yaml

# Start LiteLLM router
exec litellm --config /app/utility/litellm.config.yaml


