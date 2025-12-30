# Contributing to ZipAct

Thank you for your interest in contributing to ZipAct! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, OS, etc.)

### Suggesting Enhancements

For feature requests or enhancements:
- Check existing issues first
- Provide clear use case
- Explain expected benefits

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Test thoroughly
5. Commit with clear messages (`git commit -m 'Add AmazingFeature'`)
6. Push to branch (`git push origin feature/AmazingFeature`)
7. Open a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/ZipAct.git
cd ZipAct

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
python test_setup.py
```

## Code Style

- Follow PEP 8
- Use type hints where possible
- Add docstrings for public methods
- Keep functions focused and modular

## Testing

Before submitting PR:
- Run `python test_setup.py`
- Test with at least one environment (ALFWorld recommended)
- Verify no breaking changes to existing agents

## Areas for Contribution

### High Priority
- **More Environments**: Add WebShop, SciWorld implementations
- **Evaluation**: More comprehensive metrics
- **Documentation**: Examples, tutorials

### Medium Priority
- **Optimization**: Caching, parallelization
- **Agents**: New baseline implementations
- **Prompts**: Improved templates, few-shot examples

### Nice to Have
- **Visualization**: Result plotting tools
- **CLI**: Better command-line interface
- **Tests**: Unit tests for core components

## Questions?

Open an issue or discussion. We're happy to help!

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
