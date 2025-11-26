# Contributing to ComfyPoe

Thanks for your interest in contributing!

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/mrf/ComfyPoe.git
   ```
3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Setup

For testing without ComfyUI, run the unit tests:

```bash
pytest tests/
```

For integration testing, symlink or copy to your ComfyUI installation:

```bash
ln -s /path/to/ComfyPoe /path/to/ComfyUI/custom_nodes/ComfyPoe
```

## Making Changes

1. Create a branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes

3. Run tests:
   ```bash
   pytest tests/
   ```

4. Commit with a descriptive message

5. Push and open a pull request

## Code Style

- Follow existing code patterns
- Use type hints where practical
- Keep functions focused and small

## Adding New Models

To add support for a new Poe image model, add it to the `POE_IMAGE_MODELS` list in `poe_image_nodes.py`.

## Reporting Issues

When reporting bugs, please include:
- ComfyUI version
- Python version
- Full error traceback
- Minimal workflow to reproduce
