
# Contributing to Financial Advisory Bot

Thank you for your interest in contributing! This guide will help you set up the project, follow coding standards, and submit your contributions smoothly.

---

## 1. Fork and Clone the Repository

1. Fork the repository by clicking the **Fork** button on GitHub.  
2. Clone your fork locally:

```bash
git clone https://github.com/your-username/financial-advisory-bot.git
cd financial-advisory-bot
```

*Replace `your-username` with your GitHub username.*

---

## 2. Set Up the Development Environment

Make sure you have Python 3.8 or above installed.

Create and activate a virtual environment (recommended):

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 3. Configure Environment Variables

Create a `.env` file by copying from the example:

```bash
copy .env.example .env      # Windows PowerShell
cp .env.example .env        # macOS/Linux
```

Then, update `.env` with your API keys and other required secrets.

---

## 4. Code Style Guidelines

- Follow PEP8 for Python code style.
- Use descriptive variable and function names.
- Keep your code modular and readable.
- Use tools like `flake8` or `black` to lint and format your code.

---

## 5. Branch Naming Conventions

Use clear, descriptive branch names prefixed by the work type:

- `feature/` for new features
- `bugfix/` for bug fixes
- `docs/` for documentation changes
- `refactor/` for code restructuring without changing functionality

Example:

```bash
git checkout -b feature/add-forecasting-model
```

---

## 6. Pull Request Submission Process

- Fork and clone the repo.
- Create a branch following the naming conventions.
- Make your changes with meaningful commit messages.
- Push your branch to your fork.
- Open a Pull Request (PR) to the main repo's `main` branch.
- Fill out the PR description clearly explaining your changes.
- Respond to review comments and update your PR if needed.

---

## 7. Adding Yourself as a Contributor

We use the [all-contributors](https://allcontributors.org/) bot to recognize contributions.

After your PR is merged, comment on your PR or issue with:

```bash
@all-contributors please add @your-username for code, doc
```

---

## 8. Linking CONTRIBUTING.md in README

Please ensure the README contains:

```markdown
## Contributing

Please read our [CONTRIBUTING.md](CONTRIBUTING.md) to get started contributing to this project.

Thank you for helping improve the Financial Advisory Bot! 🎉

This project is part of GSSoC'25 — Good luck to all participants!
```
