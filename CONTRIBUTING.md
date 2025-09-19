# Contributing to Self-Organizing Maps Explorer

We welcome contributions to the Self-Organizing Maps Explorer project! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Development Setup

### Running Tests
```bash
python demo.py  # Test basic functionality
streamlit run streamlit_app.py  # Test UI
```

### Code Style
- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Add docstrings to all functions and classes
- Keep functions focused and modular

## Types of Contributions

### Bug Reports
- Use the issue template
- Include steps to reproduce
- Provide system information
- Include error messages and stack traces

### Feature Requests
- Describe the use case
- Explain the expected behavior
- Consider implementation complexity

### Code Contributions
- Create a feature branch from main
- Write tests for new functionality
- Update documentation as needed
- Ensure all tests pass

## Pull Request Process

1. Update the README.md with details of changes if applicable
2. Update the requirements.txt if you add dependencies
3. Increase version numbers following semantic versioning
4. Ensure your code follows the existing style
5. Write clear commit messages

## Code Review Guidelines

- Be respectful and constructive
- Focus on the code, not the person
- Explain the reasoning behind suggestions
- Test the changes locally when possible

## Areas for Contribution

### High Priority
- Additional SOM algorithms (GSOM, GHSOM)
- More evaluation metrics
- Performance optimizations
- Better error handling

### Medium Priority
- Additional datasets
- Export functionality (PDF reports)
- Batch processing capabilities
- API endpoints

### Low Priority
- Additional visualization options
- Theme customization
- Internationalization

## Documentation

- Keep README.md up to date
- Document new features
- Include code examples
- Update docstrings

## Questions?

Feel free to open an issue for questions or join discussions in existing issues.

Thank you for contributing! 🎉
