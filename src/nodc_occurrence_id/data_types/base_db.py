import datetime
import pathlib
import uuid
from abc import ABC, abstractmethod
from typing import Any, Type
import numpy as np

import polars as pl
import sqlalchemy as sa
import sqlalchemy.orm as orm

from nodc_occurrence_id import event

Base = orm.declarative_base()


class DataTypeDatabaseTable:
    id: orm.Mapped[int] = orm.mapped_column(primary_key=True, autoincrement=True)
    uuid: orm.Mapped[str] = orm.mapped_column(unique=True, nullable=False)
    create_time: orm.Mapped[datetime.datetime] = orm.mapped_column(insert_default=sa.func.now(), nullable=False)
    update_time: orm.Mapped[datetime.datetime] = orm.mapped_column(insert_default=sa.func.now(), nullable=False)
    all_cols: orm.Mapped[str] = orm.mapped_column(unique=True, index=True)
    # sample_date: orm.Mapped[str]
    # sample_time: orm.Mapped[str]
    # sample_year: orm.Mapped[int]
    # sample_month: orm.Mapped[int]
    # sample_day: orm.Mapped[int]

    def __repr__(self) -> str:
        return ' : '.join([f'{key}={value}' for key, value in self.fields_data.items()])

    @property
    def fields(self) -> list[str]:
        inst = sa.inspect(self)
        return [c_attr.key for c_attr in inst.mapper.column_attrs]

    @property
    def columns(self) -> list[str]:
        inst = sa.inspect(self)
        return [c_attr.key for c_attr in inst.mapper.column_attrs
                if c_attr.key not in ['id',
                                      'uuid',
                                      'all_cols',
                                      'create_time',
                                      'update_time']]

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
    def mandatory_columns(self) -> None:
        return

    def add_all_cols_field(self) -> None:
        self.all_cols = '<>'.join(self.data.values())


class DataTypeMatching(ABC):
    def __init__(self, obj: DataTypeDatabaseTable, match_obj: DataTypeDatabaseTable) -> None:
        self.obj: DataTypeDatabaseTable = obj
        self.match_obj: DataTypeDatabaseTable = match_obj

    def __repr__(self):
        match_str = f'matching {self.match_uuid}' if self.is_valid_match() else 'no match'
        diff_str_list = [f'{key}: {item["match_value"]} (old) -> {item["value"]} (new)' for key, item in self.diff_columns.items()]
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
    _name = 'occurrence_id'  # This is the name of the id column

    # def __init__(self, db_directory: pathlib.Path | str | None = None) -> None:
    def __init__(self, db_path: pathlib.Path | str) -> None:

        self._cls: Type[DataTypeDatabaseTable] = self.cls
        self._cls_obj: DataTypeDatabaseTable = self.cls()

        self.columns = self._cls_obj.columns[:]
        self.mandatory_columns = self._cls_obj.mandatory_columns[:]

        self.db_path = db_path

        self._initiate_database()
        self._create_database()

        db_path = pathlib.Path(
            r"C:/mw/git/nodc_config/nodc_occurrence_id/occurrence_id.sqlite")
        self._db_df = pl.DataFrame()
        if db_path.exists():
            user_conn = 'sqlite:///' + str(db_path)
            self._db_df = pl.read_database_uri(
                "SELECT * FROM plankton_imaging",
                user_conn,
            )

    def _initiate_database(self):
        self._db = sa.create_engine(f'sqlite:///{self.db_path}')
        self.Session = orm.sessionmaker(bind=self._db)

    def _create_database(self) -> None:
        Base.metadata.create_all(self._db)

    @property
    def id_column(self) -> str:
        return self._name

    @property
    def temp_id_str_column(self) -> str:
        return f'_{self._name}_str'

    def search_db(self, **kwargs):
        with self.Session() as session:
            query = session.query(self._cls)
            for key, value in kwargs.items():
                query = query.filter(getattr(self._cls, key) == str(value))
            return query.all()

    def _get_table_obj(self, series: dict, include_uuid: bool = False, include_all_cols: bool = False) -> DataTypeDatabaseTable | None:
        """Returns the Database table object if all mandatory columns are present"""
        ddict = self.filter_dict(series, include_uuid=include_uuid, include_all_cols=include_all_cols)
        missing_cols = [col for col in self.mandatory_columns if not ddict[col].strip()]
        if missing_cols:
            event.post_event('missing_mandatory_columns',
                             dict(
                                 missing_columns=missing_cols,
                                 temp_id=series[self.temp_id_str_column]
                             ),
                            )
            return
        obj = self._cls(**ddict)
        return obj

    def _get_uuid_in_db_from_table_object(self, obj: DataTypeDatabaseTable) -> str | None:
        """
        Search for match in database and returns the corresponding uuid.
        A match is found if all values of self.columns match.
        Returns None if no match found.
        """
        # ans = self._db_df.filter(pl.col("all_cols") == obj.all_cols)
        # if ans.is_empty():
        #     return
        # return ans["uuid"][0]
        with self.Session() as session:
            res = session.query(self._cls).filter(self._cls.all_cols == obj.all_cols).first()
            if not res:
                return
            return res.uuid

    def _get_suggestion_in_db(self, obj: DataTypeDatabaseTable) -> DataTypeMatching | None:
        """Returns the best matches found in database."""
        for i in range(len(obj.columns)):
            cols = obj.columns[:]
            cols.pop(i)
            result = self._get_db_match_for_columns(obj, cols)
            for res_obj in result:
                matching = self.matching_cls(obj, res_obj)
                print(f"{matching.is_valid_match()=}")
                if matching.is_valid_match():
                    return matching

    def _get_db_match_for_columns(self,
                                  obj: DataTypeDatabaseTable,
                                  cols: list[str]) -> list[Type[DataTypeDatabaseTable]]:
        with self.Session() as session:
            query = session.query(self._cls)
            for col in cols:
                value = getattr(obj, col)
                query = query.filter(getattr(self._cls, col) == value)
            result = query.all()
            return result

    def _add_objs_to_db(self, objs: list[DataTypeDatabaseTable]) -> None:
        """Adds all given objects to the database"""
        print()
        print(f"_add_objs_to_db: {objs=}")
        print()
        if not objs:
            return
        with self.Session() as session:
            session.add_all(objs)
            session.commit()

    def _update_db_from_match_obj(self, valid_matches: list[DataTypeMatching]) -> None:
        """Adds all given objects to the database"""
        if not valid_matches:
            return
        update_time = datetime.datetime.now(datetime.UTC)
        with self.Session() as session:
            # updated = []
            for match in valid_matches:
                obj = session.query(self._cls).filter(self._cls.uuid == match.match_obj.uuid).first()
                for col, value in match.obj.data.items():
                    if col in ['id', 'uuid', 'create_time']:
                        continue
                    setattr(obj, col, value)
                obj.add_all_cols_field()
                obj.update_time = update_time
                # updated.append(obj)
            session.commit()

    def _add_temp_id_str_column(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.concat_str(self.columns, separator="<>").alias(self.temp_id_str_column)
        )

    def _add_ids_to_df(self, df: pl.DataFrame, id_mapper: dict[str, str]):
        return df.with_columns(
            pl.col(self.temp_id_str_column)
            .replace_strict(id_mapper, default="")
            .alias(self.id_column)
        )

    def add_uuid_to_data_and_database(self, df: pl.DataFrame, add_if_valid: bool = False) -> pl.DataFrame:
        """Adds uuid to dataframe for all rows that have match in database.
        If not match in database a new id is created and added to dataframe and database.
        Option to also add if 'self.is_valid_match' if True (set flag add_if_valid=True)"""
        import time
        t0 = time.time()
        expr = None
        if self.id_column not in df.columns:
            df = df.with_columns(
                pl.lit("").alias(self.id_column)
            )

        # Preparing data
        df = self._add_temp_id_str_column(df=df)
        mask = pl.lit(True)
        for col in self.mandatory_columns:
            mask = mask & (pl.col(col) != "")
        missing_mandatory = len(df.filter(~mask))

        data = df.filter(mask)
        tot_occurrences = len(list(data.group_by(self.temp_id_str_column)))

        tot_nr_perfect_matches = 0
        tot_nr_new = 0
        id_mapper = {}

        objs_to_add_to_db = []
        valid_matches_to_update_in_database: list[DataTypeMatching] = []
        valid_not_added: list[DataTypeMatching] = []

        i = 0
        for (temp_id_str, ), red_df in data.group_by(self.temp_id_str_column):
            series_dict = red_df[0].to_dicts()[0]
            obj = self._get_table_obj(series_dict)
            if not obj:
                continue
            obj.add_all_cols_field()
            if i and not i % 1000:
                self._post_event_progress(i, tot_occurrences)
            _id = self._get_uuid_in_db_from_table_object(obj)
            if _id:
                """ Perfect match in database. Add database UUID to dataframe"""
                id_mapper[temp_id_str] = _id
                tot_nr_perfect_matches += 1
            else:
                valid_match = self._get_suggestion_in_db(obj)
                if not valid_match:
                    """No perfect match or valid suggestions in database"""
                    _id = str(uuid.uuid4())
                    obj.uuid = _id
                    objs_to_add_to_db.append(obj)
                    id_mapper[temp_id_str] = _id
                    tot_nr_new += 1
                else:
                    if add_if_valid:
                        valid_matches_to_update_in_database.append(valid_match)
                        id_mapper[temp_id_str] = valid_match.match_uuid
                    else:
                        valid_not_added.append(valid_match)
            i += 1

        df = self._add_ids_to_df(df, id_mapper)
        self._add_objs_to_db(objs_to_add_to_db)
        self._update_db_from_match_obj(valid_matches_to_update_in_database)
        print(f"{time.time()-t0=}")

        if missing_mandatory:
            event.post_event('result',
                             dict(
                                 name='missing_mandatory',
                                 value=missing_mandatory,
                                 msg=f'Mandatory columns missing in {missing_mandatory} rows'
                             )
                             )

        if tot_nr_perfect_matches:
            event.post_event('result',
                             dict(
                                 name='nr_perfect_match',
                                 value=tot_nr_perfect_matches,
                                 msg=f'Adding {tot_nr_perfect_matches} occurence_id(s) from perfect match in database'
                             )
                             )

        if valid_matches_to_update_in_database:
            event.post_event('result',
                             dict(
                                 name='valid_added',
                                 value=valid_matches_to_update_in_database,
                                 msg=f'Adding {len(valid_matches_to_update_in_database)} '
                                     f'occurence_id(s) from VALID match in database. '
                                     f'Database is updated!'
                             )
                             )

        if valid_not_added:
            event.post_event('result',
                             dict(
                                 name='valid_not_added',
                                 value=valid_not_added,
                                 msg=f'Found {len(valid_not_added)} '
                                     f'VALID occurence_id match(es)in database but did not add! '
                                     f'Set add_if_valid=True if you want to add them'
                             )
                             )

        if tot_nr_new:
            event.post_event('result',
                             dict(
                                 name='nr_new_ids',
                                 value=tot_nr_new,
                                 msg=f'{tot_nr_new} new occurens_id(s) added to data and database'
                             )
                             )
        return df

    def _post_event_progress(self, current: int, total: int) -> None:
        event.post_event('progress',
                         dict(
                             total=total,
                             # total=tot_nr_occurrences,
                             current=current,
                             title='Checking occurrence id'
                         )
                         )

    def filter_dict(self, data: dict, include_uuid: bool = False, include_all_cols: bool = False) -> dict:
        new_data = {}
        columns = self.columns
        if include_uuid:
            columns.append('uuid')
        if include_all_cols:
            columns.append('all_cols')
        for col in columns:
            from_col = col
            if col == 'uuid':
                from_col = self.id_column
            value = data.get(from_col, '')
            new_data[col] = value
        return new_data
