import pathlib
from typing import Type

from nodc_config import Config

from nodc_occurrence_id import utils

from ..occurrence import OccurrencesDatabase
from .harbour_seal import HarbourSealOccurrencesDatabase
from .plankton_imaging import PlanktonImagingOccurrencesDatabase
from .zoobenthos import ZoobenthosOccurrencesDatabase


def get_database_path(nodc_conf: Config, data_type: str) -> pathlib.Path | None:
    root_directory = nodc_conf.get_directory("nodc_occurrence_id")
    if not root_directory:
        root_directory = nodc_conf.root_dir / "nodc_occurrence_id"
        root_directory.mkdir(parents=True)
    return root_directory / f"occurrence_id_{data_type}.txt"


def get_occurrence_database_path_for_data_type(
    nodc_conf: Config, data_type: str
) -> pathlib.Path:
    db_path = get_database_path(nodc_conf, data_type)
    return db_path


def get_occurrence_database_for_data_type(
    nodc_conf: Config, data_type: str
) -> OccurrencesDatabase:
    cls = get_databases().get(data_type.lower())
    if not cls:
        return
    path = get_occurrence_database_path_for_data_type(nodc_conf, data_type)
    if not path:
        raise FileNotFoundError("No ")
    path.parent.mkdir(parents=True, exist_ok=True)
    return cls(nodc_conf, path)


def get_databases() -> dict[str, Type[OccurrencesDatabase]]:
    dbs = {}
    for data_type, cls in utils.get_all_class_children(OccurrencesDatabase).items():
        dbs[data_type] = cls
    return dbs


def get_database_names(nodc_conf: Config) -> list[str]:
    return list(get_databases())
