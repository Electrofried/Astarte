# Contributing to Project Astarte

Thank you for your interest in contributing to Project Astarte! This document provides guidelines and information for contributors.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful and constructive in your interactions with other contributors.

## How to Contribute

1. Fork the repository
2. Create a new branch for your feature or fix
3. Make your changes
4. Write or update tests as needed
5. Update documentation if necessary
6. Submit a pull request

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/[Electrofried]/astarte.git
cd astarte
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `interface.py` - Core model implementation
- `web_interface.py` - Gradio web interface
- `requirements.txt` - Project dependencies
- `run.bat`/`run.sh` - Launch scripts

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and modular
- Comment complex algorithms or mathematical operations

## Testing

- Write tests for new features
- Ensure existing tests pass
- Test both normal and edge cases
- Include test data when necessary

## Documentation

- Update README.md for significant changes
- Document new features and API changes
- Include docstrings for new functions and classes
- Add comments explaining complex logic

## Pull Request Process

1. Update the README.md with details of changes if needed
2. Update the requirements.txt if you add dependencies
3. Ensure your code follows the project's style guidelines
4. Write clear commit messages
5. Link any related issues in your pull request description

## License

By contributing to Project Astarte, you agree that your contributions will be licensed under the GNU Affero General Public License v3.0.

## Questions or Need Help?

Feel free to open an issue for:
- Bug reports
- Feature requests
- Documentation improvements
- General questions

Thank you for contributing to Project Astarte!