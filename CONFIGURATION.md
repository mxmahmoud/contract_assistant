# Configuration Management (Container-first)

This guide focuses on running the app in containers by default (Ollama + TEI + LiteLLM + Streamlit), while still supporting an online OpenAI mode.

## Overview

The configuration system is built on the following principles:

- **Single Source of Truth**: All configuration values are defined in one place
- **Environment-Based**: Support for development, staging, and production environments
- **12-Factor App**: Follows the 12-factor app methodology for configuration
- **Type Safety**: Uses Pydantic for validation and type checking
- **Flexibility**: Easy to switch between OpenAI and local LLM providers

## Configuration Structure

### Core Configuration (`ca_core/config.py`)

The main configuration class `Settings` provides:

- **Environment Configuration**: Development, staging, production modes
- **LLM Provider Configuration**: OpenAI vs. local services
- **Service Configuration**: Hosts, ports, and URLs for all services
- **Model Configuration**: Model names and aliases
- **Vector Store Configuration**: Database types and persistence settings
- **Docker Configuration**: Network names and service configurations

### Configuration Utilities (`utility/config_utils.py`)

Helper functions for:

- Generating configuration files
- Validating settings
- Exporting configurations
- Managing environment variables

### Command-Line Interface (`utility/config_utils.py`)

CLI tool for:

- Generating configuration files
- Validating settings
- Showing configuration summaries
- Managing different configuration types

## Quick Start (Containers)

### 1) Prepare `.env`

```bash
# Generate all configuration files
make config-generate

# Or use the script directly
python utility/config_utils.py render-config utility/litellm.config.template.yaml utility/litellm.config.yaml
```

Copy and adjust the example `.env`:

```bash
# Copy the example file
cp env.example .env

# Edit the file
nano .env
```

### 2) Validate Configuration (optional)

```bash
# Validate your configuration
make config-validate

# Or use the script
python utility/config_utils.py validate
```

## Configuration Options (Key Variables)

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ENVIRONMENT` | Application environment | `development` | No |
| `DEBUG` | Enable debug mode | `false` | No |
| `LLM_PROVIDER` | LLM provider (openai/local) | `local` | No |
| `OPENAI_API_KEY` | OpenAI API key | - | Yes (if using OpenAI) |
| `APP_PORT` | Main application port | `8501` | No |
| `LITELLM_PORT` | LiteLLM service port | `4000` | No |
| `OLLAMA_PORT` | Ollama service port | `11434` | No |
| `TEI_PORT` | TEI service port | `8504` | No |

### LLM Provider Configuration

#### OpenAI Mode (`LLM_PROVIDER=openai`)

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key_here
OPENAI_LLM_MODEL=gpt-4o-mini
OPENAI_EMBEDDINGS_MODEL=text-embedding-3-small
```

#### Local Mode (`LLM_PROVIDER=local`)

```bash
LLM_PROVIDER=local
LOCAL_LLM_MODEL=ollama/llama3.2:3b
LOCAL_EMBEDDINGS_MODEL=intfloat/multilingual-e5-large-instruct
LOCAL_LLM_ALIAS=gpt-local
LOCAL_EMBEDDINGS_ALIAS=local-embeddings
```

### Service Configuration (Defaults for Compose)

```bash
# Main Application
APP_HOST=0.0.0.0
APP_PORT=8501

# LiteLLM Proxy
LITELLM_HOST=litellm
LITELLM_PORT=4000

# Ollama
OLLAMA_HOST=ollama
OLLAMA_PORT=11434

# Text Embeddings Inference
TEI_HOST=tei
TEI_PORT=8504
```

## Docker & Persistence

The Docker Compose files automatically use these environment variables and bind-mount host directories for persistence:

```yaml
services:
  app:
    ports:
      - "${APP_PORT:-8501}:${APP_PORT:-8501}"
    networks:
      - ${DOCKER_NETWORK_NAME:-contract-net}
```

### Persistence Paths

- Vector store (Chroma): `data/chroma_data`
- Contract registry: `data/contracts/<contract_id>/`
  - PDF, `meta.json`, `entities.json`
- Ollama data: `data/weights/ollama_data`
- TEI data: `data/weights/tei_data`

## Configuration Management Commands

### Makefile Targets

```bash
# Configuration help
make config-help

# Generate all configurations
make config-generate

# Validate configuration
make config-validate

# Show configuration summary
make config-summary

# Generate specific configurations
make config-env
make config-litellm
make config-docker
```

### Direct Script Usage

```bash
# Generate .env file
python utility/config_utils.py --generate-env

# Validate configuration
python utility/config_utils.py --validate

# Show summary
python utility/config_utils.py --summary

# Generate all files with verbose output
python utility/config_utils.py --generate-all --verbose
```

## Environment-Specific Configurations

### Development (Local containers)

```bash
ENVIRONMENT=development
DEBUG=true
LLM_PROVIDER=local  # Use local services
```

### Production

```bash
ENVIRONMENT=production
DEBUG=false
LLM_PROVIDER=openai  # Use OpenAI API
```

### Staging

```bash
ENVIRONMENT=staging
DEBUG=false
LLM_PROVIDER=openai  # Use OpenAI API
```

## Validation and Error Handling

The configuration system includes comprehensive validation:

- **Required Fields**: Checks for required API keys and settings
- **Port Validation**: Ensures ports are in valid ranges (1024-65535)
- **Environment Warnings**: Warns about debug mode in production
- **Directory Checks**: Verifies persistence directories exist

## Migration from Old Configuration

If you have existing configuration files:

1. **Backup** your current configuration
2. **Generate** new configuration files: `make config-generate`
3. **Copy** your existing values to the new `.env` file
4. **Validate** the configuration: `make config-validate`
5. **Test** the application with new configuration

## Best Practices

### 1. Environment Separation

- Use different `.env` files for different environments
- Never commit `.env` files to version control
- Use `.env.example` as a template

### 2. Security

- Store sensitive values (API keys) in environment variables
- Use different API keys for different environments
- Rotate API keys regularly

### 3. Configuration Management

- Use the configuration management tools provided
- Validate configuration before deployment
- Document environment-specific requirements

### 4. Docker

- Use environment variables in Docker Compose
- Mount configuration files as volumes when needed
- Use Docker secrets for sensitive data in production

## Troubleshooting

### Common Issues

1. **Missing API Key**: Ensure `OPENAI_API_KEY` is set when using OpenAI
2. **Port Conflicts**: Check that ports are not already in use
3. **Service Unreachable**: Verify hostnames and ports in Docker network
4. **Configuration Validation Errors**: Use `make config-validate` to identify issues

### Debug Mode

Enable debug mode to see detailed configuration information:

```bash
DEBUG=true
```

### Configuration Summary

View current configuration:

```bash
make config-summary
```

## Advanced Configuration

### Custom Model Configurations

You can customize model configurations by modifying the environment variables:

```bash
# Custom OpenAI models
OPENAI_LLM_MODEL=gpt-4
OPENAI_EMBEDDINGS_MODEL=text-embedding-ada-002
# Custom local models (served via Ollama/TEI)
LOCAL_LLM_MODEL=ollama/llama3.2:3b
LOCAL_EMBEDDINGS_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Vector Store Configuration

Support for different vector store types:

```bash
VECTOR_STORE_TYPE=chroma  # Default
VECTOR_STORE_TYPE=postgres
VECTOR_STORE_TYPE=sqlite
```

### Service Discovery

For advanced deployments, you can configure service discovery:

```bash
# Use service discovery
LITELLM_HOST=litellm-service.internal
OLLAMA_HOST=ollama-service.internal
TEI_HOST=tei-service.internal
```

### LiteLLM Health Checks

Container health checks use the readiness endpoint per LiteLLM docs. For ongoing model health, enable background checks in the Litellm config template:

```yaml
general_settings:
  background_health_checks: true
  health_check_interval: 300
  health_check_details: false
```

See: https://docs.litellm.ai/docs/proxy/health

## Contributing

When adding new configuration options:

1. Add the field to the `Settings` class in `ca_core/config.py`
2. Include proper validation and documentation
3. Update the configuration utilities
4. Add examples to this documentation
5. Update the environment file template

## Support

For configuration issues:

1. Check the validation output: `make config-validate`
2. Review the configuration summary: `make config-summary`
3. Check the logs for specific error messages
4. Verify environment variable values
5. Ensure Docker services are running (for local mode)
