import os


def ensure_dir(path: str) -> None:
    """
    SUMMARY
    -------
    Ensure that a directory exists; if it doesn't, create it.

    PARAMETERS
    ----------
    path : str
        Path to the directory to ensure.
    """
    os.makedirs(path, exist_ok=True)
