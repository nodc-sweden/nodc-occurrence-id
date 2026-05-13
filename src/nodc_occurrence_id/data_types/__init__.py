import pathlib
from typing import Type

from nodc_occurrence_id import utils

from ..occurrence import OccurrencesDatabase
from .plankton_imaging import PlanktonImagingOccurrencesDatabase
from .zoobenthos import ZoobenthosOccurrencesDatabase


def get_database_path(data_type: str) -> pathlib.Path | None:
    if not utils.DATABASE_DIRECTORY:
        return
    return utils.DATABASE_DIRECTORY / f"occurrence_id_{data_type}.txt"


def get_occurrence_database_path_for_data_type(data_type: str) -> pathlib.Path | None:
    db_path = get_database_path(data_type)
    return db_path


def get_occurrence_database_for_data_type(data_type: str) -> OccurrencesDatabase | None:
    cls = get_databases().get(data_type.lower())
    if not cls:
        return
    path = get_occurrence_database_path_for_data_type(data_type)
    if not path:
        return
    return cls(path)


def get_databases() -> dict[str, Type[OccurrencesDatabase]]:
    dbs = {}
    for data_type, cls in utils.get_all_class_children(OccurrencesDatabase).items():
        dbs[data_type] = cls
    return dbs


def get_database_names() -> list[str]:
    return list(get_databases())
