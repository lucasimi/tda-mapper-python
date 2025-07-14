"""
This module is the entry point for running the application.
"""

from tdamapper.app import main


def run() -> None:
    """
    Run the application.
    """
    main()


if __name__ in {"__main__", "__mp_main__", "tdamapper._run_app"}:
    run()
