"""
ComfyUI Poe Image Nodes
=======================
Custom nodes for image generation and editing via Poe's OpenAI-compatible API.

Nodes included:
- PoeImageEdit: Edit images with text prompts
- PoeImageGenerate: Generate images from text
- PoeImageVariation: Create variations of images
- PoeMultiImageEdit: Edit with multiple reference images

Usage:
1. Get your API key from https://poe.com/api_key
2. Add a Poe node to your workflow
3. Enter your API key and configure settings
"""

import subprocess
import sys


def _ensure_dependencies():
    """Auto-install dependencies on first load if missing."""
    try:
        import openai
        import requests
    except ImportError:
        print("[ComfyPoe] Installing required dependencies...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "openai>=1.0.0", "requests>=2.28.0"]
        )
        print("[ComfyPoe] Dependencies installed successfully!")


_ensure_dependencies()

from .poe_image_nodes import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

WEB_DIRECTORY = "./web"
