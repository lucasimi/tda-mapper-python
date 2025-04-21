# Contributing

Thank you for considering contributing to **tda-mapper**! We welcome
contributions from everyone. To ensure a smooth and productive experience,
please follow the guidelines below.

## Reporting Issues

Found a bug or have a feature request? Here's how to get started:

1. **Search for Existing Issues**. Check the
[issue tracker](https://github.com/lucasimi/tda-mapper-python/issues) to see if
the issue has already been reported.

2. **Open a New Issue**. If you can't find any open issue related to your
problem, create a new issue and provide as much detail as possible, including
steps to reproduce the problem or context for the feature request.

## Proposing Changes

Contribution comes in multiple forms and we value your feedback! If you have
ideas or suggestions to improve the project, feel free to start a discussion in
the issue tracker.

## Contributing Code

We encourage contributions to fix bugs, enhance documentation, improve testing,
or address open issues. Contributions involving new algorithms or data
structures are welcome but must first be discussed in an issue, especially when
it comes to performance critical parts. Please include supporting theoretical
or experimental evidence for such contributions.

NB: Contributions from bots or fully automated AI agents are discouraged and
will be discarded.

Follow these steps to contribute:

1. **Fork the Repository**.
    Start by forking the repository and cloning it locally:

    ```bash
    git clone https://github.com/lucasimi/tda-mapper-python.git
    ```

2. **Set Up Your Development Environment**.
    Navigate to the project directory, create a virtual environment and install
    the dev dependencies:

    ```bash
    cd tda-mapper-python
    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    pip install -e .[dev]
    ```

3. **Make Your Changes**.
    Work on a dedicated branch for your feature or bug fix:

    ```bash
    git checkout -b your-feature-branch
    ```

    Write clear, well-documented code. Add tests for any new functionality to
    ensure correctness. All performance critical commits should be covered by
    a benchmark test to avoid performance regressions. Unit tests should follow
    the naming convention `test_unit_*.py`, and benchmark tests should follow
    the naming convention `test_bench_*.py`.

4. **Run Tests**.
    Ensure your changes pass all tests before committing. We use `unittest` as
    test framework:

    ```bash
    python -m unittest discover -s tests -p 'test_*.py'
    ```

    Before each commit make sure to check code coverage:

    ```bash
    coverage run --source=src -m unittest discover -s tests -p 'test_*.py'
    ```

5. **Commit and Push Your Changes**.
    Commit your changes with a descriptive message:

    ```bash
    git commit -m "Fixes bug with feature X"
    git push origin your-feature-branch
    ```

6. **Submit a Pull Request**.
    Once your changes are ready, tested with adequate coverage, and documented,
    open a pull request (PR) on the repository's GitHub page. Be sure to:

    - Provide a clear title and description for your PR.

    - Reference any related issues by using #issue-number.

    All pull requests will be reviewed by at least one core contributor. They
    may be subject to rejection or may require changes before merging.

### Backward Compatibility

We prioritize backward compatibility according to semantic versioning. Code
contributions introducing breaking changes must be discussed and decided in a
related issue. Every deprecation must be clearly warned to users calling
deprecated APIs. New releases are decided from core contributors, based on the
addition of new features, bug fixes, and deprecations. Deprecated APIs are
removed whenever required by a new jump in major version.

### Code Style

We follow [PEP 8](https://peps.python.org/pep-0008/) for Python code style.
You can run a linter to check your code. The dev dependencies of **tda-mapper**
include `black` and `isort` to help you manage that.

```bash
black .
```

```bash
isort .
```

### Documentation

Ensure that new features and APIs are documented in the code and that
documentation is correctly generated. Documentation is generated from
docstrings using sphinx, and docstrings should follow Sphinx format.
To build the documentation locally:

```bash
cd docs
make clean
make html
```

## Thank You!

We appreciate your contributions and are excited to collaborate with you on
making **tda-mapper** better! If you have any questions, feel free to reach
out via the issue tracker.
