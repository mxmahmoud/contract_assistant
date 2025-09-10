# utility/config_utils.py
"""
Configuration utilities for the Contract Assistant application.
Provides validation and utility functions for configuration management.
"""

import os
import re
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from utility.config import settings, BASE_DIR
from ca_core.exceptions import ConfigurationError, ValidationError


def substitute_env_vars(template_text: str, mapping: Optional[Dict[str, str]] = None, strict: bool = False) -> Tuple[str, set]:
    """
    Substitute ${VAR} and ${VAR:-default} tokens in template text.
    
    Args:
        template_text: Template text with ${VAR} placeholders
        mapping: Environment variables mapping (defaults to os.environ)
        strict: If True, raise error on missing variables
        
    Returns:
        Tuple of (rendered text, set of missing variables)
    """
    if mapping is None:
        mapping = dict(os.environ)
    
    # Find ${...} tokens
    token_re = re.compile(r'\$\{([^}]+)\}')
    missing = set()
    
    def replace_token(match):
        inner = match.group(1)
        
        # Support ${VAR:-default} syntax
        if ':-' in inner:
            var, default = inner.split(':-', 1)
            var = var.strip()
            val = mapping.get(var)
            if val is None or val == '':
                return default
            return val
        else:
            var = inner.strip()
            val = mapping.get(var)
            if val is None:
                missing.add(var)
                if strict:
                    raise ValidationError(f"Missing required variable: {var}")
                return ''
            return val
    
    rendered = token_re.sub(replace_token, template_text)
    
    if missing and strict:
        raise ValidationError(f"Missing required variables: {', '.join(sorted(missing))}")
    
    return rendered, missing


def render_config_template(template_path: Path, output_path: Optional[Path] = None, strict: bool = False) -> str:
    """
    Render a configuration template file with environment variable substitution.
    
    Args:
        template_path: Path to template file
        output_path: Optional output path. If None, returns rendered content
        strict: If True, fail on missing variables
        
    Returns:
        Rendered configuration content
    """
    if not template_path.exists():
        raise ConfigurationError(f"Template file not found: {template_path}")
    
    try:
        template_content = template_path.read_text(encoding='utf-8')
        rendered_content, missing = substitute_env_vars(template_content, strict=strict)
        
        if missing:
            print(f"Warning: Missing variables replaced with empty string: {', '.join(sorted(missing))}", file=sys.stderr)
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered_content, encoding='utf-8')
            print(f"Rendered configuration saved to: {output_path}")
        
        return rendered_content
        
    except Exception as e:
        raise ConfigurationError(f"Failed to render template: {e}")


def validate_configuration() -> Dict[str, Any]:
    """
    Validate the current configuration and return validation results.
    
    Returns:
        Dictionary containing validation results and any issues found
    """
    issues = []
    warnings = []
    
    # Check required API key for OpenAI
    if settings.LLM_PROVIDER == "openai" and not settings.OPENAI_API_KEY:
        issues.append("OPENAI_API_KEY is required when using OpenAI provider")
    
    # Check if ports are in valid range
    for port_name, port_value in [
        ("APP_PORT", settings.APP_PORT),
        ("LITELLM_PORT", settings.LITELLM_PORT),
        ("OLLAMA_PORT", settings.OLLAMA_PORT),
        ("TEI_PORT", settings.TEI_PORT),
    ]:
        if not (1024 <= port_value <= 65535):
            issues.append(f"{port_name} must be between 1024 and 65535")
    
    # Check if directories exist
    if not Path(settings.CHROMA_PERSIST_DIR).exists():
        warnings.append(f"ChromaDB persist directory does not exist: {settings.CHROMA_PERSIST_DIR}")
    
    # Check environment-specific warnings
    if settings.ENVIRONMENT == "production" and settings.DEBUG:
        warnings.append("Debug mode is enabled in production environment")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "provider": settings.LLM_PROVIDER,
        "environment": settings.ENVIRONMENT,
    }


def get_config_summary() -> Dict[str, Any]:
    """
    Get a comprehensive summary of the current configuration.
    
    Returns:
        Dictionary containing configuration summary
    """
    return {
        "environment": {
            "name": settings.ENVIRONMENT,
            "debug": settings.DEBUG,
        },
        "llm_provider": {
            "type": settings.LLM_PROVIDER,
            "is_local": settings.is_local_mode,
        },
        "models": {
            "llm": settings.llm_model,
            "embeddings": settings.embeddings_model,
        },
        "services": {
            "app": f"{settings.APP_HOST}:{settings.APP_PORT}",
            "litellm": settings.litellm_url,
            "ollama": settings.ollama_url,
            "tei": settings.tei_url,
        },
        "vector_store": {
            "type": settings.VECTOR_STORE_TYPE,
            "chroma_dir": settings.CHROMA_PERSIST_DIR,
        },
        "docker": {
            "network": settings.DOCKER_NETWORK_NAME,
        },
    }


def check_service_health(service_url: str, timeout: int = 5) -> bool:
    """
    Check if a service is healthy by making an HTTP request.
    
    Args:
        service_url: URL of the service to check
        timeout: Request timeout in seconds
        
    Returns:
        True if service is healthy, False otherwise
    """
    import requests
    try:
        response = requests.get(f"{service_url}/health", timeout=timeout)
        return response.status_code == 200
    except:
        return False


def main():
    """Main CLI entry point for configuration management."""
    parser = argparse.ArgumentParser(
        description="Contract Assistant Configuration Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Render LiteLLM config from template
  python utility/config_utils.py render-config utility/litellm.config.template.yaml utility/litellm.config.yaml
  
  # Validate current configuration
  python utility/config_utils.py validate
  
  # Show configuration summary
  python utility/config_utils.py summary
  
  # Check if services are healthy
  python utility/config_utils.py check-services
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    render_parser = subparsers.add_parser('render-config', help='Render configuration from template')
    render_parser.add_argument('template', type=Path, help='Template file path')
    render_parser.add_argument('output', type=Path, help='Output file path (use "-" for stdout)')
    render_parser.add_argument('--strict', action='store_true', help='Fail if any variables are missing')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate configuration')
    
    # Summary command
    summary_parser = subparsers.add_parser('summary', help='Show configuration summary')
    
    # Check services command
    check_parser = subparsers.add_parser('check-services', help='Check if services are healthy')
    
    args = parser.parse_args()
    
    try:
        if args.command == 'render-config':
            if args.output.name == '-':
                content = render_config_template(args.template, strict=args.strict)
                sys.stdout.write(content)
            else:
                render_config_template(args.template, args.output, strict=args.strict)
                
        elif args.command == 'validate':
            result = validate_configuration()
            print("\nConfiguration Validation:")
            print("=" * 50)
            
            if result['valid']:
                print("✅ Configuration is valid!")
            else:
                print("❌ Configuration has issues:")
                for issue in result['issues']:
                    print(f"  - {issue}")
            
            if result['warnings']:
                print("\n⚠️  Warnings:")
                for warning in result['warnings']:
                    print(f"  - {warning}")
            
            print(f"\nProvider: {result['provider']}")
            print(f"Environment: {result['environment']}")
            
            sys.exit(0 if result['valid'] else 1)
            
        elif args.command == 'summary':
            summary = get_config_summary()
            print("\nConfiguration Summary:")
            print("=" * 50)
            
            for section, data in summary.items():
                print(f"\n{section.replace('_', ' ').title()}:")
                for key, value in data.items():
                    print(f"  {key.replace('_', ' ').title()}: {value}")
                    
        elif args.command == 'check-services':
            if not settings.is_local_mode:
                print("Service health checks are only available in local mode")
                sys.exit(0)
            
            print("\nChecking service health...")
            print("=" * 50)
            
            services = {
                'LiteLLM': settings.litellm_url,
                'Ollama': settings.ollama_url,
                'TEI': settings.tei_url,
            }
            
            all_healthy = True
            for name, url in services.items():
                healthy = check_service_health(url)
                status = "✅ Healthy" if healthy else "❌ Unhealthy"
                print(f"{name}: {status} ({url})")
                all_healthy = all_healthy and healthy
            
            sys.exit(0 if all_healthy else 1)
            
        else:
            parser.print_help()
            
    except ConfigurationError as e:
        print(f"\nConfiguration Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValidationError as e:
        print(f"\nValidation Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

