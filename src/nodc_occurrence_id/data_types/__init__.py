import logging
import pathlib
from typing import Type

from nodc_occurrence_id import utils
from .base import OccurrencesDatabase
from .plankton_imaging import PlanktonImagingOccurrencesDatabase
from .zoobenthos import ZoobenthosOccurrencesDatabase

logger = logging.getLogger(__name__)


def get_database_path(name: str) -> pathlib.Path | None:
    # if name not in get_database_names():
    #     raise FileNotFoundError(f'No config file with name "{name}" exists')
    if not utils.DATABASE_DIRECTORY:
        return
    return utils.DATABASE_DIRECTORY / name


# def get_database_path(name: str) -> pathlib.Path:
#     # if name not in get_database_names():
#     #     raise FileNotFoundError(f'No config file with name "{name}" exists')
#     if utils.DATABASE_DIRECTORY:
#         path = utils.DATABASE_DIRECTORY / name
#         if path.exists():
#             return path
#     temp_path = utils.TEMP_DATABASE_DIRECTORY / name
#     if temp_path.exists():
#         return temp_path
#     update_database_file(temp_path)
#     return temp_path


# def update_database_file(path: pathlib.Path) -> None:
#     path.parent.mkdir(exist_ok=True, parents=True)
#     url = utils.DATABASES_URL + path.name
#     try:
#         res = requests.get(url, verify=ssl.CERT_NONE)
#         if res.status_code == 404:
#             return
#         with open(path, 'wb') as fid:
#             for chunk in res.iter_content(chunk_size=128):
#                 fid.write(chunk)
#             logger.info(f'Database file "{path.name}" updated from {url}')
#     except requests.exceptions.ConnectionError:
#         logger.warning(f'Connection error. Could not update database file {path.name}')
#         raise


# def update_database_files() -> None:
#     """Downloads database files from github"""
#     for dtype in get_databases():
#         name = get_database_name_for_data_type(dtype)
#         target_path = utils.TEMP_DATABASE_DIRECTORY / name
#         update_database_file(target_path)


def get_occurrence_database_path_for_data_type(data_type: str) -> pathlib.Path | None:
    name = get_database_name_for_data_type(data_type)
    db_path = get_database_path(name)
    return db_path


def get_occurrence_database_for_data_type(data_type: str) -> OccurrencesDatabase | None:
    cls = get_databases().get(data_type.lower())
    if not cls:
        return
    path = get_occurrence_database_path_for_data_type(data_type)
    if not path:
        return
    return cls(path)


def get_database_name_for_data_type(data_type: str) -> str:
    return f'occurrence_id.sqlite'
    # return f'occurrence_id_{data_type.lower()}.sqlite'


def get_databases() -> dict[str, Type[OccurrencesDatabase]]:
    dbs = {}
    for data_type, cls in utils.get_all_class_children(OccurrencesDatabase).items():
        dbs[data_type] = cls
    return dbs


def get_database_names() -> list[str]:
    return list(get_databases())
