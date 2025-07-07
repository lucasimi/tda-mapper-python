# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Notebook preprocessing
import glob
import os
import subprocess

NOTEBOOK_DIR = os.path.join(os.path.dirname(__file__), "notebooks")


def convert_py_to_ipynb():
    py_files = glob.glob(os.path.join(NOTEBOOK_DIR, "*.py"))
    for py_file in py_files:
        ipynb_file = py_file.replace(".py", ".ipynb")
        subprocess.run(
            ["jupytext", "--to", "ipynb", "--output", ipynb_file, py_file], check=True
        )
        subprocess.run(
            [
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--inplace",
                ipynb_file,
            ],
            check=True,
        )


convert_py_to_ipynb()

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "tda-mapper"
copyright = "2024, Luca Simi"
author = "Luca Simi"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_logo = "logos/tda-mapper-logo-horizontal.png"
html_theme_options = {
    "sticky_navigation": True,
    "vcs_pageview_mode": "blob",
}
html_context = {
    "display_github": True,
    "github_user": "lucasimi",
    "github_repo": "tda-mapper-python",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
}
