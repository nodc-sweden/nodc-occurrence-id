import pathlib
from typing import Type

from nodc_occurrence_id import utils

from ..occurrence import OccurrencesDatabase
from .harbour_seal import HarbourSealOccurrencesDatabase
from .plankton_imaging import PlanktonImagingOccurrencesDatabase
from .zoobenthos import ZoobenthosOccurrencesDatabase


def get_database_path(data_type: str,
                      root_directory: pathlib.Path | None = None) -> (
        pathlib.Path | None):
    if not root_directory:
        root_directory = utils.DATABASE_DIRECTORY
    if not root_directory:
        return
    return root_directory / f"occurrence_id_{data_type}.txt"


def get_occurrence_database_path_for_data_type(data_type: str,
                                               root_directory: pathlib.Path | None = None) -> pathlib.Path | None:
    db_path = get_database_path(
        data_type,
        root_directory=root_directory
    )
    return db_path


def get_occurrence_database_for_data_type(data_type: str,
                                          root_directory: pathlib.Path | None = None) -> (
        OccurrencesDatabase |
                                                                 None):
    cls = get_databases().get(data_type.lower())
    if not cls:
        return
    path = get_occurrence_database_path_for_data_type(data_type,
                                                      root_directory=root_directory)
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    return cls(path)


def get_databases() -> dict[str, Type[OccurrencesDatabase]]:
    dbs = {}
    for data_type, cls in utils.get_all_class_children(OccurrencesDatabase).items():
        dbs[data_type] = cls
    return dbs


def get_database_names() -> list[str]:
    return list(get_databases())
