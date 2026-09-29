# Contributing

Issues and pull requests are welcome.

- Open an issue to report a bug or discuss a larger change before starting on it.
- Keep pull requests focused, and add or update tests in `tests/` for behaviour changes.
- Run the CPU test suite before opening a pull request:

  ```bash
  pip install -e ".[test]"
  pytest -q
  ```

- The rendered prompt text is part of each model; a change to it changes the model's
  output and needs a clear reason and matching canary evidence.

A Developer Certificate of Origin (DCO) sign-off is not required. By contributing,
you agree that your contribution is licensed under the repository's Apache-2.0 license.
