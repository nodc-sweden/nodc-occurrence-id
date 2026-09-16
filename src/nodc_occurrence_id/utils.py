import hashlib
import pathlib

from nodc_config import Config


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


def get_database_hashes(nodc_conf: Config) -> dict[str, str]:
    hashes = dict()
    for path in nodc_conf.get_directory("nodc_occurrence_id").iterdir():
        if not path.suffix == ".txt":
            continue
        if not path.name.startswith("occurrence_id_"):
            continue
        hashes[path.name] = _get_hash_of_file(path)
    return hashes
