import logging
import os
import pathlib

logger = logging.getLogger(__name__)

CONFIG_ENV = 'NODC_CONFIG'

CONFIG_SUBDIRECTORY = 'nodc_occurrence_id'
CONFIG_FILE_NAMES = []


DATABASE_DIRECTORY = None
if os.getenv(CONFIG_ENV) and pathlib.Path(os.getenv(CONFIG_ENV)).exists():
    DATABASE_DIRECTORY = pathlib.Path(os.getenv(CONFIG_ENV)) / CONFIG_SUBDIRECTORY


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

