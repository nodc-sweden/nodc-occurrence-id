import hashlib
import os
import pathlib

CONFIG_ENV = "NODC_CONFIG"

CONFIG_SUBDIRECTORY = "nodc_occurrence_id"
CONFIG_FILE_NAMES = []

home = pathlib.Path.home()
OTHER_CONFIG_SOURCES = [
    home / "NODC_CONFIG",
    home / ".NODC_CONFIG",
    home / "nodc_config",
    home / ".nodc_config",
]


def get_user_given_config_dir() -> pathlib.Path | None:
    path = pathlib.Path(os.getcwd()) / "config_directory.txt"
    if not path.exists():
        return
    with open(path) as fid:
        config_path = fid.readline().strip()
        if not config_path:
            return
        config_path = pathlib.Path(config_path)
        if not config_path.exists():
            return
        return config_path


def get_all_class_children_list(cls):
    if not cls.__subclasses__():
        return []
    children = []
    for c in cls.__subclasses__():
        children.append(c)
        children.extend(get_all_class_children_list(c))
    return children


def get_all_class_children(cls):
    mapping = dict()
    for c in get_all_class_children_list(cls):
        mapping[c.data_type.lower()] = c
    return mapping


def _get_hash_of_file(path: pathlib.Path) -> str:
    with open(str(path), "rb") as f:
        return hashlib.file_digest(f, hashlib.sha256).hexdigest()


def get_database_hashes() -> dict[str, str]:
    hashes = dict()
    for path in DATABASE_DIRECTORY.iterdir():
        if not path.suffix == ".txt":
            continue
        if not path.name.startswith("occurrence_id_"):
            continue
        hashes[path.name] = _get_hash_of_file(path)
    return hashes


CONFIG_DIRECTORY = get_user_given_config_dir()
if not CONFIG_DIRECTORY:
    if os.getenv(CONFIG_ENV) and pathlib.Path(os.getenv(CONFIG_ENV)).exists():
        CONFIG_DIRECTORY = pathlib.Path(os.getenv(CONFIG_ENV))
    else:
        for directory in OTHER_CONFIG_SOURCES:
            if directory.exists():
                CONFIG_DIRECTORY = directory
                break
DATABASE_DIRECTORY = CONFIG_DIRECTORY / CONFIG_SUBDIRECTORY
