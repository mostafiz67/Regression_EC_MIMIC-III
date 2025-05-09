from pathlib import Path


def ensure_dir(path: Path) -> Path:
    if path.exists():
        if path.is_dir():
            return path
        else:
            raise FileExistsError("Specified path exists and is a file, not directory")
    for parent in path.parents:
        if not parent.exists():
            raise FileNotFoundError(
                f"The parent directory {parent} does not exist. You have mis-specified a path constant or argument."
            )
    # exist_ok=True in rare case of TOCTOU / parallelism
    path.mkdir(exist_ok=True, parents=False)
    return path
