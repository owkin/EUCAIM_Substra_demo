from pathlib import Path
from . import logger


def extract_data(data_path: Path):
    for file in data_path.iterdir():
        file_found = False
        if file.is_file():
            (data_path / file.stem).mkdir(exist_ok=True)
            file.rename(file.parent / file.stem / file.name)
            file_found = True

    if not file_found:
        logger.info(f"{data_path.name} seems already extracted. Skipping data extraction.")
