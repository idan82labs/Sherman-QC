#!/usr/bin/env python3
"""
Generate OpenAPI documentation in JSON and YAML formats.

Usage:
    python scripts/generate_openapi.py
"""

import sys
import json
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

def main():
    # Import app - avoid importing modules that require external dependencies
    try:
        # Mock the missing modules to allow import
        import importlib.util

        # Create a minimal mock for fitz (PyMuPDF)
        class MockFitz:
            pass
        sys.modules['fitz'] = MockFitz

        from fastapi.openapi.utils import get_openapi
        from server import app

    except Exception as e:
        print(f"Warning: Could not import server directly: {e}")
        print("Generating minimal OpenAPI spec...")

        # Generate a minimal spec
        openapi_spec = {
            "openapi": "3.1.0",
            "info": {
                "title": "Sherman Scan QC System API",
                "description": "AI-Powered Quality Control for Sheet Metal Parts",
                "version": "1.0.0"
            },
            "paths": {}
        }

        output_dir = Path(__file__).parent.parent / "docs" / "api"
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "openapi.json", "w") as f:
            json.dump(openapi_spec, f, indent=2)

        print(f"Minimal OpenAPI spec generated at: {output_dir / 'openapi.json'}")
        return

    # Generate full OpenAPI spec
    openapi_spec = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=app.openapi_tags
    )

    # Output directory
    output_dir = Path(__file__).parent.parent / "docs" / "api"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write JSON
    json_path = output_dir / "openapi.json"
    with open(json_path, "w") as f:
        json.dump(openapi_spec, f, indent=2)
    print(f"Generated: {json_path}")

    # Try to write YAML if pyyaml is available
    try:
        import yaml
        yaml_path = output_dir / "openapi.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(openapi_spec, f, default_flow_style=False, sort_keys=False)
        print(f"Generated: {yaml_path}")
    except ImportError:
        print("Note: Install 'pyyaml' to generate YAML format")

    # Print summary
    print(f"\nOpenAPI {openapi_spec['openapi']} spec generated")
    print(f"Title: {openapi_spec['info']['title']}")
    print(f"Version: {openapi_spec['info']['version']}")
    print(f"Endpoints: {len(openapi_spec['paths'])}")


if __name__ == "__main__":
    main()
