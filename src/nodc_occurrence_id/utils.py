import pathlib

THIS_DIR = pathlib.Path(__file__).parent
DATABASES_DIR = THIS_DIR / 'DATABASES'


def get_database_path(database_name: str) -> pathlib.Path:
    if not database_name.endswith('.sqlite'):
        database_name = database_name + '.sqlite'
    return DATABASES_DIR / database_name