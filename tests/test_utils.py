"""Tests for utility functions (no torch dependency)."""

import base64
import io
import re

import pytest
from PIL import Image


# Copy the pure functions here to test in isolation
def extract_image_urls(text: str) -> list[str]:
    """Extract image URLs from markdown or plain text response."""
    markdown_urls = re.findall(r"!\[.*?\]\((https?://[^\s\)]+)\)", text)
    if markdown_urls:
        return markdown_urls

    standalone_urls = re.findall(
        r'(https?://[^\s<>"\']+\.(?:png|jpg|jpeg|gif|webp)(?:\?[^\s<>"\']*)?)', text, re.IGNORECASE
    )
    if standalone_urls:
        return standalone_urls

    any_urls = re.findall(r'(https://[^\s<>"\']+)', text)
    return any_urls


def image_to_base64(pil_img: Image.Image, format: str = "PNG") -> str:
    """Encode PIL Image to base64 string."""
    buffered = io.BytesIO()
    pil_img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


class TestExtractImageUrls:
    def test_markdown_image_url(self):
        text = "Here is your image: ![Generated Image](https://example.com/image.png)"
        urls = extract_image_urls(text)
        assert urls == ["https://example.com/image.png"]

    def test_multiple_markdown_urls(self):
        text = """
        ![Image 1](https://example.com/img1.png)
        Some text
        ![Image 2](https://example.com/img2.jpg)
        """
        urls = extract_image_urls(text)
        assert len(urls) == 2
        assert "https://example.com/img1.png" in urls
        assert "https://example.com/img2.jpg" in urls

    def test_standalone_image_url(self):
        text = "Your image is at https://cdn.example.com/output.png please download it"
        urls = extract_image_urls(text)
        assert "https://cdn.example.com/output.png" in urls

    def test_url_with_query_params(self):
        text = "![img](https://example.com/image.png?token=abc123&size=large)"
        urls = extract_image_urls(text)
        assert urls == ["https://example.com/image.png?token=abc123&size=large"]

    def test_no_urls(self):
        text = "Sorry, I could not generate an image."
        urls = extract_image_urls(text)
        assert urls == []

    def test_fallback_to_any_https(self):
        text = "Image available at https://api.poe.com/files/12345"
        urls = extract_image_urls(text)
        assert "https://api.poe.com/files/12345" in urls


class TestImageToBase64:
    def test_encode_rgb_image(self):
        img = Image.new("RGB", (100, 100), color="red")
        b64 = image_to_base64(img)
        assert isinstance(b64, str)
        assert len(b64) > 0
        assert b64.startswith("iVBOR")  # PNG header

    def test_encode_rgba_image(self):
        img = Image.new("RGBA", (50, 50), color=(255, 0, 0, 128))
        b64 = image_to_base64(img, format="PNG")
        assert isinstance(b64, str)
        assert len(b64) > 0

    def test_roundtrip(self):
        img = Image.new("RGB", (10, 10), color="blue")
        b64 = image_to_base64(img)
        decoded = base64.b64decode(b64)
        restored = Image.open(io.BytesIO(decoded))
        assert restored.size == (10, 10)
