from pathlib import Path

PROJECT_NAME = "bike_sharing_regressor_model"

STRUCTURE = {
    "configs": [
        "data.yml",
        "model.yml",
        "train.yml",
    ],
    PROJECT_NAME: {
        "config": ["__init__.py", "settings.py"],
        "data": ["__init__.py", "load.py", "preprocess.py"],
        "features": ["__init__.py", "build_features.py"],
        "models": ["__init__.py", "train.py", "evaluate.py", "predict.py"],
        "pipelines": ["__init__.py", "training_pipeline.py"],
        "utils": ["__init__.py", "logger.py"],
        "__init__.py": None,
    },
    "tests": [
        "test_data.py",
        "test_features.py",
        "test_models.py",
    ],
    "scripts": [
        "train.py",
        "predict.py",
    ],
}

ROOT_FILES = [
    "README.md",
    ".gitignore",
    ".env",
    "pyproject.toml",
]


def create_file(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)


def create_structure(base_path: Path, structure: dict):
    for name, content in structure.items():
        path = base_path / name

        if isinstance(content, dict):
            path.mkdir(exist_ok=True)
            create_structure(path, content)

        elif isinstance(content, list):
            path.mkdir(exist_ok=True)
            for file in content:
                create_file(path / file)

        elif content is None:
            create_file(path)


def main():
    base_path = Path(PROJECT_NAME)
    base_path.mkdir(exist_ok=True)

    # Root files
    for file in ROOT_FILES:
        create_file(base_path / file)

    # Folder structure
    create_structure(base_path, STRUCTURE)

    print("✅ ML project structure created successfully!")


if __name__ == "__main__":
    main()
