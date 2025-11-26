"""
ComfyUI Poe Image Nodes
Custom nodes for image generation and editing via Poe's OpenAI-compatible API.
"""

import base64
import io
import re

import numpy as np
import requests
import torch
from PIL import Image

try:
    import openai
except ImportError:
    openai = None


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert ComfyUI IMAGE tensor to PIL Image."""
    # ComfyUI images are [B, H, W, C] float32 in range [0, 1]
    img_np = tensor.cpu().numpy().squeeze()
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(img_np)


def pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    """Convert PIL Image to ComfyUI IMAGE tensor."""
    img_np = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(img_np).unsqueeze(0)


def image_to_base64(pil_img: Image.Image, format: str = "PNG") -> str:
    """Encode PIL Image to base64 string."""
    buffered = io.BytesIO()
    pil_img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def extract_image_urls(text: str) -> list[str]:
    """Extract image URLs from markdown or plain text response."""
    # Match markdown image syntax: ![alt](url)
    markdown_urls = re.findall(r"!\[.*?\]\((https?://[^\s\)]+)\)", text)
    if markdown_urls:
        return markdown_urls

    # Match standalone URLs ending in image extensions
    standalone_urls = re.findall(
        r'(https?://[^\s<>"\']+\.(?:png|jpg|jpeg|gif|webp)(?:\?[^\s<>"\']*)?)', text, re.IGNORECASE
    )
    if standalone_urls:
        return standalone_urls

    # Match any https URL (some APIs return URLs without extensions)
    any_urls = re.findall(r'(https://[^\s<>"\']+)', text)
    return any_urls


def fetch_image_from_url(url: str, timeout: int = 60) -> Image.Image:
    """Download image from URL and return as PIL Image."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content))


POE_IMAGE_MODELS = [
    "GPT-Image-1",
    "DALL-E-3",
    "Imagen-4",
    "FLUX-pro-1.1",
    "FLUX-schnell",
    "Playground-v3",
    "Recraft-V3",
    "Ideogram-v2",
    "SD3.5-Large",
    "SD3.5-Large-Turbo",
]

ASPECT_RATIOS = ["1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16", "auto"]
QUALITY_LEVELS = ["low", "medium", "high"]


class PoeImageEdit:
    """
    Edit images using Poe's chat completions API with image models.
    Supports GPT-Image-1, DALL-E-3, Imagen-4, FLUX, and other image models.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "Edit this image to..."}),
                "api_key": ("STRING", {"default": ""}),
            },
            "optional": {
                "model": (POE_IMAGE_MODELS, {"default": "GPT-Image-1"}),
                "aspect": (ASPECT_RATIOS, {"default": "auto"}),
                "quality": (QUALITY_LEVELS, {"default": "high"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "raw_response")
    FUNCTION = "edit_image"
    CATEGORY = "api/poe"

    def edit_image(
        self,
        image: torch.Tensor,
        prompt: str,
        api_key: str,
        model: str = "GPT-Image-1",
        aspect: str = "auto",
        quality: str = "high",
        seed: int = -1,
    ) -> tuple[torch.Tensor, str]:
        if openai is None:
            raise ImportError("openai package is required. Install with: pip install openai")

        if not api_key:
            raise ValueError("API key is required")

        # Convert input tensor to base64
        pil_img = tensor_to_pil(image)
        img_b64 = image_to_base64(pil_img)

        # Build request
        client = openai.OpenAI(api_key=api_key, base_url="https://api.poe.com/v1")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ],
            }
        ]

        extra_body = {}
        if aspect != "auto":
            extra_body["aspect"] = aspect
        if quality:
            extra_body["quality"] = quality
        if seed >= 0:
            extra_body["seed"] = seed

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            extra_body=extra_body if extra_body else None,
            stream=False,
        )

        content = response.choices[0].message.content

        # Extract and fetch image
        urls = extract_image_urls(content)
        if not urls:
            raise ValueError(f"No image URL found in response: {content}")

        result_img = fetch_image_from_url(urls[0])
        output_tensor = pil_to_tensor(result_img)

        return (output_tensor, content)


class PoeImageGenerate:
    """
    Generate images from text using Poe's chat completions API.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A beautiful landscape..."}),
                "api_key": ("STRING", {"default": ""}),
            },
            "optional": {
                "model": (POE_IMAGE_MODELS, {"default": "GPT-Image-1"}),
                "aspect": (ASPECT_RATIOS, {"default": "1:1"}),
                "quality": (QUALITY_LEVELS, {"default": "high"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "raw_response")
    FUNCTION = "generate_image"
    CATEGORY = "api/poe"

    def generate_image(
        self,
        prompt: str,
        api_key: str,
        model: str = "GPT-Image-1",
        aspect: str = "1:1",
        quality: str = "high",
        seed: int = -1,
        negative_prompt: str = "",
    ) -> tuple[torch.Tensor, str]:
        if openai is None:
            raise ImportError("openai package is required. Install with: pip install openai")

        if not api_key:
            raise ValueError("API key is required")

        client = openai.OpenAI(api_key=api_key, base_url="https://api.poe.com/v1")

        # Build full prompt with negative prompt if provided
        full_prompt = prompt
        if negative_prompt:
            full_prompt = f"{prompt}\n\nAvoid: {negative_prompt}"

        messages = [{"role": "user", "content": full_prompt}]

        extra_body = {}
        if aspect != "auto":
            extra_body["aspect"] = aspect
        if quality:
            extra_body["quality"] = quality
        if seed >= 0:
            extra_body["seed"] = seed

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            extra_body=extra_body if extra_body else None,
            stream=False,
        )

        content = response.choices[0].message.content

        urls = extract_image_urls(content)
        if not urls:
            raise ValueError(f"No image URL found in response: {content}")

        result_img = fetch_image_from_url(urls[0])
        output_tensor = pil_to_tensor(result_img)

        return (output_tensor, content)


class PoeImageVariation:
    """
    Create variations of an input image using Poe's image models.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {"default": ""}),
            },
            "optional": {
                "model": (POE_IMAGE_MODELS, {"default": "GPT-Image-1"}),
                "variation_strength": (["subtle", "moderate", "strong"], {"default": "moderate"}),
                "aspect": (ASPECT_RATIOS, {"default": "auto"}),
                "quality": (QUALITY_LEVELS, {"default": "high"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "style_prompt": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "raw_response")
    FUNCTION = "create_variation"
    CATEGORY = "api/poe"

    def create_variation(
        self,
        image: torch.Tensor,
        api_key: str,
        model: str = "GPT-Image-1",
        variation_strength: str = "moderate",
        aspect: str = "auto",
        quality: str = "high",
        seed: int = -1,
        style_prompt: str = "",
    ) -> tuple[torch.Tensor, str]:
        if openai is None:
            raise ImportError("openai package is required. Install with: pip install openai")

        if not api_key:
            raise ValueError("API key is required")

        pil_img = tensor_to_pil(image)
        img_b64 = image_to_base64(pil_img)

        # Build variation prompt based on strength
        strength_prompts = {
            "subtle": "Create a very similar image with only minor variations in details",
            "moderate": "Create a variation of this image, keeping the main subject and composition but with noticeable changes",
            "strong": "Create a reimagined version of this image with significant creative changes while maintaining the core concept",
        }

        base_prompt = strength_prompts.get(variation_strength, strength_prompts["moderate"])
        if style_prompt:
            base_prompt = f"{base_prompt}. Style: {style_prompt}"

        client = openai.OpenAI(api_key=api_key, base_url="https://api.poe.com/v1")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": base_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ],
            }
        ]

        extra_body = {}
        if aspect != "auto":
            extra_body["aspect"] = aspect
        if quality:
            extra_body["quality"] = quality
        if seed >= 0:
            extra_body["seed"] = seed

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            extra_body=extra_body if extra_body else None,
            stream=False,
        )

        content = response.choices[0].message.content

        urls = extract_image_urls(content)
        if not urls:
            raise ValueError(f"No image URL found in response: {content}")

        result_img = fetch_image_from_url(urls[0])
        output_tensor = pil_to_tensor(result_img)

        return (output_tensor, content)


class PoeMultiImageEdit:
    """
    Edit images using multiple reference images (e.g., for style transfer or combining elements).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "Combine these images..."}),
                "api_key": ("STRING", {"default": ""}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                "model": (POE_IMAGE_MODELS, {"default": "GPT-Image-1"}),
                "aspect": (ASPECT_RATIOS, {"default": "auto"}),
                "quality": (QUALITY_LEVELS, {"default": "high"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "raw_response")
    FUNCTION = "edit_with_reference"
    CATEGORY = "api/poe"

    def edit_with_reference(
        self,
        image: torch.Tensor,
        prompt: str,
        api_key: str,
        reference_image: torch.Tensor | None = None,
        model: str = "GPT-Image-1",
        aspect: str = "auto",
        quality: str = "high",
        seed: int = -1,
    ) -> tuple[torch.Tensor, str]:
        if openai is None:
            raise ImportError("openai package is required. Install with: pip install openai")

        if not api_key:
            raise ValueError("API key is required")

        # Convert main image
        pil_img = tensor_to_pil(image)
        img_b64 = image_to_base64(pil_img)

        # Build content with images
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
        ]

        # Add reference image if provided
        if reference_image is not None:
            ref_pil = tensor_to_pil(reference_image)
            ref_b64 = image_to_base64(ref_pil)
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ref_b64}"}}
            )

        client = openai.OpenAI(api_key=api_key, base_url="https://api.poe.com/v1")

        messages = [{"role": "user", "content": content}]

        extra_body = {}
        if aspect != "auto":
            extra_body["aspect"] = aspect
        if quality:
            extra_body["quality"] = quality
        if seed >= 0:
            extra_body["seed"] = seed

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            extra_body=extra_body if extra_body else None,
            stream=False,
        )

        content_response = response.choices[0].message.content

        urls = extract_image_urls(content_response)
        if not urls:
            raise ValueError(f"No image URL found in response: {content_response}")

        result_img = fetch_image_from_url(urls[0])
        output_tensor = pil_to_tensor(result_img)

        return (output_tensor, content_response)


# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "PoeImageEdit": PoeImageEdit,
    "PoeImageGenerate": PoeImageGenerate,
    "PoeImageVariation": PoeImageVariation,
    "PoeMultiImageEdit": PoeMultiImageEdit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PoeImageEdit": "Poe Image Edit",
    "PoeImageGenerate": "Poe Image Generate",
    "PoeImageVariation": "Poe Image Variation",
    "PoeMultiImageEdit": "Poe Multi-Image Edit",
}
