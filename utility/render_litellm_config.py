# utility/render_litellm_config.py
"""
A self-contained script to render the LiteLLM configuration template.
This script is designed to be run in a minimal environment
"""

import os
import re
import sys
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple, Set


def substitute_env_vars(template_text: str, mapping: Optional[Dict[str, str]] = None, strict: bool = False) -> Tuple[str, Set[str]]:
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
    
    token_re = re.compile(r'\$\{([^}]+)\}')
    missing: Set[str] = set()
    
    def replace_token(match):
        inner = match.group(1)
        
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
                    raise ValueError(f"Missing required variable: {var}")
                return ''
            return val
    
    rendered = token_re.sub(replace_token, template_text)
    
    if missing and strict:
        raise ValueError(f"Missing required variables: {', '.join(sorted(missing))}")
    
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
        raise FileNotFoundError(f"Template file not found: {template_path}")
    
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
        raise RuntimeError(f"Failed to render template: {e}")


def main():
    """Main CLI entry point for configuration rendering."""
    parser = argparse.ArgumentParser(
        description="Render LiteLLM configuration from a template.",
        epilog="Example: python render_litellm_config.py template.yaml output.yaml"
    )
    
    parser.add_argument('template', type=Path, help='Template file path')
    parser.add_argument('output', type=Path, help='Output file path (use "-" for stdout)')
    parser.add_argument('--strict', action='store_true', help='Fail if any variables are missing')
    
    args = parser.parse_args()
    
    try:
        if args.output.name == '-':
            content = render_config_template(args.template, strict=args.strict)
            sys.stdout.write(content)
        else:
            render_config_template(args.template, args.output, strict=args.strict)
            
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
