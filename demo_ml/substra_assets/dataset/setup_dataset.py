from pathlib import Path


def extract_data(data_path: Path):
    for file in data_path.iterdir():
        if file.is_file():
            (data_path / file.stem).mkdir(exist_ok=True)
            file.rename(file.parent / file.stem / file.name)
