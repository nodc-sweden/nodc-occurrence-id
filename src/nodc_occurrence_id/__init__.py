import functools
import pathlib

from nodc_occurrence_id.base import OccurrencesDatabase


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
        mapping[c.__name__.lower()] = c
    return mapping


database_mapping = get_all_class_children(OccurrencesDatabase)


def get_occurrence_database(data_type: str, db_directory: str | pathlib.Path = None) -> OccurrencesDatabase | None:
    cls = database_mapping.get(data_type.lower())
    if not cls:
        return None
    return cls(db_directory=db_directory)


# THIS_DIR = pathlib.Path(__file__).parent
# CONFIG_DIR = THIS_DIR / 'CONFIG_FILES'
#
# CONFIG_URLS = [
#     r'https://raw.githubusercontent.com/nodc-sweden/nodc-codes/main/src/nodc_codes/CONFIG_FILES/translate_codes.txt',
# ]
#
#
# @functools.cache
# def get_translate_codes_object() -> "TranslateCodes":
#     path = CONFIG_DIR / 'translate_codes.txt'
#     return TranslateCodes(path)
#
#
# def update_config_files() -> None:
#     """Downloads config files from github"""
#     try:
#         for url in CONFIG_URLS:
#             name = pathlib.Path(url).name
#             target_path = CONFIG_DIR / name
#             res = requests.get(url, verify=ssl.CERT_NONE)
#             with open(target_path, 'w', encoding='cp1252') as fid:
#                 fid.write(res.text)
#                 logger.info(f'Config file "{name}" updated from {url}')
#     except requests.exceptions.ConnectionError:
#         logger.warning('Connection error. Could not update config files!')
#         raise
