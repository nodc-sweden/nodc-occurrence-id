import pathlib
import uuid
from abc import ABC, abstractmethod
from typing import Any, Type

import pandas as pd
import sqlalchemy as sa
import sqlalchemy.orm as orm

from nodc_occurrence_id import utils, event

# from sharkadm.utils import get_export_directory
# from sharkadm import adm_logger


Base = orm.declarative_base()


class DataTypeDatabaseTable:
    id: orm.Mapped[int] = orm.mapped_column(primary_key=True, autoincrement=True)
    uuid: orm.Mapped[str] = orm.mapped_column(unique=True)
    all_cols: orm.Mapped[str] = orm.mapped_column()

    def __repr__(self) -> str:
        return ' : '.join([f'{key}={value}' for key, value in self.fields_data.items()])

    @property
    def fields(self) -> list[str]:
        inst = sa.inspect(self)
        return [c_attr.key for c_attr in inst.mapper.column_attrs]

    @property
    def columns(self) -> list[str]:
        inst = sa.inspect(self)
        return [c_attr.key for c_attr in inst.mapper.column_attrs if c_attr.key not in ['id', 'uuid', 'all_cols']]

    @property
    def fields_data(self) -> dict:
        return dict((col, getattr(self, col)) for col in self.fields)

    @property
    def data(self) -> dict:
        return dict((col, getattr(self, col)) for col in self.columns)

    @property
    def nr_columns(self) -> int:
        return len(self.columns)

    @property
    def mandatory_columns(self):
        pass

    def add_all_cols_field(self):
        self.all_cols = '<>'.join(self.data.values())


class DataTypeMatching(ABC):
    def __init__(self, obj: DataTypeDatabaseTable, match_obj: DataTypeDatabaseTable) -> None:
        self.obj = obj
        self.match_obj = match_obj

    def __repr__(self):
        match_str = f'matching {self.match_uuid}' if self.is_valid_match() else 'no match'
        diff_str_list = [f'{key}: {item["value"]} -> {item["match_value"]}' for key, item in self.diff_columns.items()]
        diff_str = '  ;  '.join(diff_str_list)
        return f'Match object ({match_str}): {diff_str}'

    @property
    def columns(self) -> list[str]:
        return self.obj.columns[:]

    @property
    def nr_columns(self) -> int:
        return self.obj.nr_columns

    @property
    def match_uuid(self) -> str:
        return self.match_obj.uuid

    @property
    def diff_columns(self) -> dict[str, dict[Any, Any]]:
        diffs = {}
        for col in self.columns:
            obj_val = getattr(self.obj, col)
            match_obj_val = getattr(self.match_obj, col)
            if obj_val == match_obj_val:
                continue
            diffs[col] = {}
            diffs[col]['value'] = obj_val
            diffs[col]['match_value'] = match_obj_val
            diffs[col]['match_uuid'] = self.match_obj.uuid
        return diffs

    @property
    def nr_diff_columns(self) -> int:
        return len(self.diff_columns)

    @property
    def percent_match(self):
        return round((self.nr_columns - self.nr_diff_columns) * 100 / self.nr_columns)

    @abstractmethod
    def is_valid_match(self) -> bool:
        """Set up rules that decides if the partial match is valid or not"""
        ...


class OccurrencesDatabase:
    data_type: str = ''
    cls: Type[DataTypeDatabaseTable] = None
    matching_cls: Type[DataTypeMatching] = None
    _name = 'unique_id_occurrence'  # This is the name of the id column

    def __init__(self, db_directory: pathlib.Path | str | None = None) -> None:

        self._cls: Type[DataTypeDatabaseTable] = self.cls
        self._cls_obj: DataTypeDatabaseTable = self.cls()

        self.columns = self._cls_obj.columns[:]
        self.mandatory_columns = self._cls_obj.mandatory_columns[:]

        name = f'occurrence_id_{self.data_type}.sqlite'

        if db_directory:
            self.db_path = pathlib.Path(db_directory) / name
        else:
            self.db_path = utils.get_database_path(name)

        self._initiate_database()
        self._create_database()

    def _initiate_database(self):
        self._db = sa.create_engine(f'sqlite:///{self.db_path}')
        self.Session = orm.sessionmaker(bind=self._db)

    def _create_database(self) -> None:
        Base.metadata.create_all(self._db)

    @property
    def id_column(self) -> str:
        return self._name

    @property
    def temp_id_column(self) -> str:
        return f'_{self._name}_str'

    def _get_df_with_no_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        self._add_temp_id_column(df=df)
        return df.drop_duplicates(self.temp_id_column)

    def _get_obj_from_series(self, series: pd.Series, include_uuid: bool = False) -> DataTypeDatabaseTable | None:
        """Returns the Database table object if all mandatory columns are present"""
        ddict = self.filter_dict(series.to_dict(), include_uuid=include_uuid)
        missing_cols = [col for col in self.mandatory_columns if not ddict[col].strip()]
        if missing_cols:
            event.post_event('missing_mandatory_columns',
                             dict(
                                 missing_columns=missing_cols,
                                 temp_id=series[self.temp_id_column]
                             ),
                            )
            return
        obj = self._cls(**ddict)
        return obj

    def _get_uuid_in_db_from_object(self, obj: DataTypeDatabaseTable) -> str | None:
        """
        Search for match in database and returns the corresponding uuid.
        A match is found if all values of self.columns match.
        Returns None if no match found.
        """
        with self.Session() as session:
            query = session.query(self._cls)
            for col in obj.columns:
                query = query.filter(getattr(self._cls, col) == getattr(obj, col))
            items = query.all()
            if items:
                return items[0].uuid

    def _add_obj(self, obj: DataTypeDatabaseTable) -> None:
        """Adds all given objects to the database"""
        with self.Session() as session:
            obj.add_all_cols_field()
            session.add(obj)
            session.commit()
            session.close()

    def _add_objs(self, objs: list[DataTypeDatabaseTable]) -> None:
        """Adds all given objects to the database"""
        if not objs:
            return
        with self.Session() as session:
            for obj in objs:
                obj.add_all_cols_field()
                session.add(obj)
            session.commit()
            session.close()

    def search_db(self, **kwargs):
        with self.Session() as session:
            query = session.query(self._cls)
            for key, value in kwargs.items():
                query = query.filter(getattr(self._cls, key) == str(value))
            return query.all()

    def add_uuid_to_dataframe(self, df: pd.DataFrame):
        """Adds uuid to dataframe for all rows that have match in database"""
        red_df = self._get_df_with_no_duplicates(df)
        for i, series in red_df.iterrows():
            obj = self._get_obj_from_series(series)
            _id = self._get_uuid_in_db_from_object(obj)
            if not _id:
                continue
            boolean = df[self.temp_id_column] == series[self.temp_id_column]
            df.loc[boolean, self.id_column] = _id

    def add_uuid_to_database_from_dataframe(self, df: pd.DataFrame):
        """Adds uuid to database from dataframe if prefect match"""
        objs_to_add_to_db = []
        red_df = self._get_df_with_no_duplicates(df)
        for i, series in red_df.iterrows():
            obj = self._get_obj_from_series(series, include_uuid=True)
            if not obj:
                continue
            if not obj.uuid:
                continue
            _id = self._get_uuid_in_db_from_object(obj)
            if _id:
                continue
            objs_to_add_to_db.append(obj)
            event.post_event('uuid_added_to_db_from_data',
                             dict(
                                 id=obj.uuid,
                             )
                             )
        self._add_objs(objs_to_add_to_db)

    def add_uuid_to_dataframe_and_database(self, df: pd.DataFrame, add_if_valid: bool = False) -> dict:
        """Adds uuid to dataframe for all rows that have match in database.
        If not match in database a new id is created and added to dataframe and database.
        Option to also add if 'is_valid_match' if True (set flag add_if_valid=True)"""
        if self.id_column not in df.columns:
            df[self.id_column] = ''
        objs_to_add_to_db = []
        all_valid_matches = {}
        red_df = self._get_df_with_no_duplicates(df)
        for i, series in red_df.iterrows():
            obj = self._get_obj_from_series(series)
            if not obj:
                continue
            _id = self._get_uuid_in_db_from_object(obj)
            if _id:
                """ Perfect match in database. Add database UUID to dataframe"""
                temp_id = series[self.temp_id_column]
                boolean = df[self.temp_id_column] == temp_id
                df.loc[boolean, self.id_column] = _id
                event.post_event('perfect_match_in_database',
                                 dict(
                                     id=_id,
                                     temp_id=temp_id
                                 )
                                 )
            else:
                match_list = self._get_suggestion_in_db(obj)
                if not match_list:
                    """No column match database"""
                    _id = str(uuid.uuid4())
                    obj.uuid = _id
                    # self._add_obj(obj)
                    objs_to_add_to_db.append(obj)
                    temp_id = series[self.temp_id_column]
                    boolean = df[self.temp_id_column] == temp_id
                    df.loc[boolean, self.id_column] = _id
                    event.post_event('no_match_in_database',
                                     dict(
                                         id=_id,
                                         temp_id=temp_id
                                     )
                                     )
                else:
                    valid_matches = [match for match in match_list if match.is_valid_match()]
                    # print()
                    # print(i)
                    # print(f'{len(valid_matches)=}')
                    # print(f'{valid_matches=}')
                    all_valid_matches[str(obj)] = valid_matches
                    # print(f'{valid_matches=}')
                    if len(valid_matches) == 1 and add_if_valid:
                        match_obj = valid_matches[0]
                        objs_to_add_to_db.append(obj)
                        boolean = df[self.temp_id_column] == series[self.temp_id_column]
                        df.loc[boolean, self.id_column] = match_obj.match_uuid

        self._add_objs(objs_to_add_to_db)
        return all_valid_matches

    def _add_temp_id_column(self, df: pd.DataFrame) -> None:
        """Values in temp id column are concatenated from values of self.columns"""
        if self.temp_id_column in df:
            return
        df[self.temp_id_column] = df[self.columns].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

    def _get_suggestion_in_db(self, obj: DataTypeDatabaseTable, return_all=False) -> list[DataTypeMatching]:
        """Returns the best matches found in database along with how many percent och the values in self.columns match. """
        result = self._get_db_match_for_columns(obj, obj.columns)
        if result:
            return [self.matching_cls(obj, result[0])]
        all_matching = []
        for i in range(len(obj.columns)):
            cols = obj.columns[:]
            cols.pop(i)
            result = self._get_db_match_for_columns(obj, cols)
            all_matching.extend([self.matching_cls(obj, mobj) for mobj in result])
        return all_matching

    def _get_db_match_for_columns(self, obj: DataTypeDatabaseTable, cols: list[str]) -> list[Type[DataTypeDatabaseTable]]:
        with self.Session() as session:
            query = session.query(self._cls)
            for col in cols:
                value = getattr(obj, col)
                query = query.filter(getattr(self._cls, col) == value)
            result = query.all()
            return result

    def filter_dict(self, data: dict, include_uuid: bool = False) -> dict:
        new_data = {}
        columns = self.columns
        if include_uuid:
            columns.append('uuid')
        for col in columns:
            from_col = col
            if col == 'uuid':
                from_col = self.id_column
            value = data.get(from_col, '')
            new_data[col] = value
        return new_data


if __name__ == '__main__':
    pass
