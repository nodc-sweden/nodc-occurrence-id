import datetime
import pathlib
import uuid
from abc import ABC, abstractmethod
from typing import Any, Type
import numpy as np

import pandas as pd
import sqlalchemy as sa
import sqlalchemy.orm as orm

from nodc_occurrence_id import event

Base = orm.declarative_base()


class DataTypeDatabaseTable:
    id: orm.Mapped[int] = orm.mapped_column(primary_key=True, autoincrement=True)
    uuid: orm.Mapped[str] = orm.mapped_column(unique=True)
    all_cols: orm.Mapped[str] = orm.mapped_column(unique=True, index=True)
    # sample_date: orm.Mapped[str]
    # sample_time: orm.Mapped[str]
    datetime_str: orm.Mapped[str]
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
        return [c_attr.key for c_attr in inst.mapper.column_attrs if c_attr.key not in ['id', 'uuid', 'all_cols',
                                                                                        # 'sample_year', 'sample_month', 'sample_day',
                                                                                        ]]

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
        self.obj: DataTypeDatabaseTable = obj
        self.match_obj: DataTypeDatabaseTable = match_obj

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

    # def _get_df_with_no_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
    #     self._add_temp_id_str_column(df=df)
    #     return df.drop_duplicates(self.temp_id_str_column)

    def _get_table_obj_from_series(self, series: pd.Series, include_uuid: bool = False, include_all_cols: bool = False) -> DataTypeDatabaseTable | None:
        """Returns the Database table object if all mandatory columns are present"""
        ddict = self.filter_dict(series.to_dict(), include_uuid=include_uuid, include_all_cols=include_all_cols)
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
        with self.Session() as session:
            res = session.query(self._cls).filter(self._cls.all_cols == obj.all_cols).first()
            if not res:
                return
            return res.uuid
            # query = session.query(self._cls)
            # for col in obj.columns:
            #     query = query.filter(getattr(self._cls, col) == getattr(obj, col))
            # items = query.all()
            # if items:
            #     return items[0].uuid

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
            session.add_all(objs)
            session.commit()

    def _update_from_match_obj(self, valid_matches: list[DataTypeMatching]) -> None:
        """Adds all given objects to the database"""
        with self.Session() as session:
            updated = []
            for match in valid_matches:
                obj = session.query(self._cls).filter(self._cls.uuid == match.match_obj.uuid).first()
                for col, value in match.obj.data.items():
                    if col in ['id', 'uuid']:
                        continue
                    setattr(obj, col, value)
                updated.append(obj)
            session.commit()

        # with self.Session() as session:
        #     for obj in objs:
        #         obj.add_all_cols_field()
        #         session.add(obj)
        #     session.commit()
        #     session.close()

    def search_db(self, **kwargs):
        with self.Session() as session:
            query = session.query(self._cls)
            for key, value in kwargs.items():
                query = query.filter(getattr(self._cls, key) == str(value))
            return query.all()

    @staticmethod
    def _get_first_series_from_dataframe(df: pd.DataFrame) -> pd.Series:
        return df.loc[list(df.index)[0]]

    def add_uuid_to_dataframe(self, df: pd.DataFrame):
        """Adds uuid to dataframe for all rows that have match in database"""
        # red_df = self._get_df_with_no_duplicates(df)
        # for i, series in red_df.iterrows():
        self._add_temp_id_str_column(df=df)
        for temp_id, red_df in df.groupby(self.temp_id_str_column):
            series = self._get_first_series_from_dataframe(red_df)
            obj = self._get_table_obj_from_series(series)
            _id = self._get_uuid_in_db_from_table_object(obj)
            if not _id:
                continue
            df.loc[red_df.index, self.id_column] = _id

    def add_uuid_to_database_from_data(self, df: pd.DataFrame):
        """Adds uuid to database from dataframe if prefect match"""
        if self.id_column not in df:
            event.post_event('no_id_column_in_data',
                             dict(msg=f'Could not add {self.id_column} to database from data')
                             )
            return

        objs_to_add_to_db = []
        self._add_temp_id_str_column(df=df)
        for _id, red_df in df.groupby(self.id_column):
            if not _id:
                event.post_event('missing_id_in_data',
                                 dict(
                                     nr_places=len(df)
                                 )
                                 )
                continue
            series = self._get_first_series_from_dataframe(red_df)
            obj = self._get_table_obj_from_series(series, include_uuid=True, include_all_cols=True)
            if not obj:
                continue
            if not obj.uuid:
                continue
            _id = self._get_uuid_in_db_from_table_object(obj)
            if _id:
                continue
            objs_to_add_to_db.append(obj)
            event.post_event('id_added_to_database_from_data',
                             dict(
                                 id=obj.uuid,
                                 nr_places=len(df),
                             )
                             )
        self._add_objs(objs_to_add_to_db)

    def _post_event_progress(self, current: int, total: int) -> None:
        event.post_event('progress',
                         dict(
                             total=total,
                             # total=tot_nr_occurrences,
                             current=current,
                             title='Checking occurrence id'
                         )
                         )

    # def _post_event_result(self, name: str, value: int | list):
    #     event.post_event('result',
    #                      dict(
    #                          name=name,
    #                          value=value
    #                      )
    #                      )

    # def _post_event_id_added_to_data_from_database(self, _id: str, temp_id_str: str, nr_places: int) -> None:
    #     event.post_event('id_added_to_data_from_database',
    #                      dict(
    #                          perfect_match_in_db=True,
    #                          id=_id,
    #                          temp_id_str=temp_id_str,
    #                          debug=True,
    #                          nr_places=nr_places,
    #                          added_to_data=True,
    #                      )
    #                      )
    #
    # def _post_event_new_id_added_to_data_and_database(self, _id: str, temp_id_str: str, nr_places: int) -> None:
    #     event.post_event('new_id_added_to_data_and_database',
    #                      dict(
    #                          no_match_in_db=True,
    #                          id=_id,
    #                          new_id=True,
    #                          temp_id_str=temp_id_str,
    #                          nr_places=nr_places,
    #                      )
    #                      )
    #
    # def _post_event_id_added_to_data_from_database_from_match(self, match_obj: DataTypeMatching, temp_id_str: str, nr_places: int) -> None:
    #     event.post_event('id_added_to_data_from_database',
    #                      dict(
    #                          valid_match=match_obj,
    #                          perfect_match_in_db=False,
    #                          diff_columns=match_obj.diff_columns,
    #                          valid_match_in_db=True,
    #                          temp_id_str=temp_id_str,
    #                          nr_places=nr_places,
    #                          added_to_data=True,
    #                      )
    #                      )
    #
    # def _post_event_valid_match_in_database(self, match_obj: DataTypeMatching, temp_id_str: str, nr_places: int) -> None:
    #     event.post_event('valid_match_in_database',
    #                      dict(
    #                          valid_match=match_obj,
    #                          perfect_match_in_db=False,
    #                          valid_match_in_db=True,
    #                          temp_id_str=temp_id_str,
    #                          nr_places=nr_places,
    #                          added_to_data=False,
    #                      )
    #                      )
    #
    # def _post_event_several_valid_matches_in_database(self, valid_matches: list[DataTypeMatching], temp_id_str: str, nr_places: int) -> None:
    #     event.post_event('several_valid_matches_in_database',
    #                      dict(
    #                          valid_matches=valid_matches,
    #                          temp_id_str=temp_id_str,
    #                          nr_places=nr_places,
    #                          warning=True,
    #                      )
    #                      )

    def add_uuid_to_data_and_database(self, df: pd.DataFrame, add_if_valid: bool = False) -> dict:
        """Adds uuid to dataframe for all rows that have match in database.
        If not match in database a new id is created and added to dataframe and database.
        Option to also add if 'self.is_valid_match' if True (set flag add_if_valid=True)"""
        import time
        # times = dict(
        #     perfect_match=0,
        #     suggestion=0,
        #     not_match_list=0,
        #     valid_matches=0,
        #     series=0,
        #     all=0,
        #     set_to_df=0,
        # )
        if self.id_column not in df.columns:
            df[self.id_column] = ''
        objs_to_add_to_db = []
        valid_matches_to_update_in_database = []
        all_valid_matches = {}
        self._add_temp_id_str_column(df=df)

        mask = np.ones(len(df)).astype(bool)
        for col in self.mandatory_columns:
            mask = mask & (df[col] != '')
        data = df[mask]
        tot_occurrences = len(data.groupby(self.temp_id_str_column))
        one_percent = int(tot_occurrences/100)
        # tot_nr_occurrences = 0
        # tot_nr_single_valid_suggestions = 0
        # # tot_nr_added_valid_suggestions = 0
        # tot_nr_not_added_valid_suggestions = 0
        # tot_nr_several_matches_not_added_valid_suggestions = 0
        tot_nr_perfect_matches = 0
        tot_nr_new = 0

        valid_not_added: list[DataTypeMatching] = []

        for i, (temp_id_str, red_df) in enumerate(data.groupby(self.temp_id_str_column)):

            # t0 = time.time()
            series = self._get_first_series_from_dataframe(red_df)
            obj = self._get_table_obj_from_series(series)
            obj.add_all_cols_field()
            # times['series'] += (time.time() - t0)
            # print(i, temp_id_str)
            # if i and not i % one_percent:
            if i and not i % 1000:
                self._post_event_progress(i, tot_occurrences)
            # print(f'a {i} ({tot_nr_occurrences}): {obj=}')
            #continue
            if not obj:
                continue
            # tot_nr_occurrences += 1
            t0 = time.time()
            _id = self._get_uuid_in_db_from_table_object(obj)
            # print(f'{_id=}')
            # times['perfect_match'] += (time.time() - t0)
            if _id:
                """ Perfect match in database. Add database UUID to dataframe"""
                df.loc[red_df.index, self.id_column] = _id
                tot_nr_perfect_matches += 1
                #self._post_event_id_added_to_data_from_database(_id, temp_id_str, nr_places)
            else:
                # t0 = time.time()
                # match_list = self._get_suggestion_in_db(obj, temp_id_str)
                valid_match = self._get_suggestion_in_db(obj)
                # print(f'{valid_match=}')
                # times['suggestion'] += (time.time() - t0)
                # if not match_list:
                if not valid_match:
                    """No perfect match or good suggestions in database"""
                    # t0 = time.time()
                    _id = str(uuid.uuid4())
                    obj.uuid = _id
                    objs_to_add_to_db.append(obj)
                    # t000 = time.time()
                    df.loc[red_df.index, self.id_column] = _id
                    # times['set_to_df'] += (time.time() - t000)
                    # self._post_event_new_id_added_to_data_and_database(_id, temp_id_str, nr_places) # Tar lååång tid
                    # times['not_match_list'] += (time.time() - t0)
                    tot_nr_new += 1
                else:
                    if add_if_valid:
                        valid_matches_to_update_in_database.append(valid_match)
                        # obj.id = valid_match.obj.id
                        # obj.uuid = valid_match.match_uuid
                        df.loc[red_df.index, self.id_column] = valid_match.match_uuid
                    else:
                        valid_not_added.append(valid_match)

                        #self._post_event_id_added_to_data_from_database_from_match(match_obj, temp_id_str, nr_places)
                    # t0 = time.time()
                    # valid_matches = [match for match in match_list if match.is_valid_match()]
                    # all_valid_matches[str(obj)] = valid_matches
                    # if len(valid_matches) == 1:
                    #     match_obj = valid_matches[0]
                    #     tot_nr_single_valid_suggestions += 1
                    #     if add_if_valid:
                    #         tot_nr_added_valid_suggestions += 1
                    #         obj.id = match_obj.match_uuid
                    #         objs_to_add_to_db.append(obj)  # Will this update?
                    #         df.loc[red_df.index, self.id_column] = match_obj.match_uuid
                    #         #self._post_event_id_added_to_data_from_database_from_match(match_obj, temp_id_str, nr_places)
                    #     else:
                    #         tot_nr_not_added_valid_suggestions += 1
                    #         pass
                    #         #self._post_event_valid_match_in_database(match_obj, temp_id_str, nr_places)
                    # elif not valid_matches:
                    #     _id = str(uuid.uuid4())
                    #     obj.uuid = _id
                    #     objs_to_add_to_db.append(obj)
                    #     df.loc[red_df.index, self.id_column] = _id
                    #     # self._post_event_new_id_added_to_data_and_database(_id, temp_id_str, nr_places) # Tar lååång tid
                    # else:
                    #     # Several valid matches
                    #     tot_nr_several_matches_not_added_valid_suggestions += 1
                    #     #self._post_event_several_valid_matches_in_database(valid_matches, temp_id_str, nr_places)
                    # times['valid_matches'] += (time.time() - t0)

        self._add_objs(objs_to_add_to_db)
        self._update_from_match_obj(valid_matches_to_update_in_database)
        # print(f'{objs_to_add_to_db=}')
        # print(f'{tot_nr_occurrences=}')
        # print(f'{tot_nr_perfect_matches=}')
        # print(f'{tot_nr_single_valid_suggestions=}')
        # print(f'{valid_not_added=}')
        # print(f'{tot_nr_not_added_valid_suggestions=}')
        # print(f'{tot_nr_several_matches_not_added_valid_suggestions=}')
        # print(f'{tot_nr_new=}')

        event.post_event('result',
                         dict(
                             name='nr_perfect_match',
                             value=tot_nr_perfect_matches,
                             msg=f'Adding {tot_nr_perfect_matches} occurence_id(s) from perfect match in database'
                         )
                         )

        event.post_event('result',
                         dict(
                             name='valid_added',
                             value=valid_matches_to_update_in_database,
                             msg=f'Adding {len(valid_matches_to_update_in_database)} occurence_id(s) from VALID match in database. Database is updates!'
                         )
                         )

        event.post_event('result',
                         dict(
                             name='valid_not_added',
                             value=valid_not_added,
                             msg=f'Found {len(valid_not_added)} VALID occurence_id match(es)in database but did not add! '
                                 f'Set add_if_valid=True if you want to add them'
                         )
                         )

        event.post_event('result',
                         dict(
                             name='nr_new_ids',
                             value=tot_nr_new,
                             msg=f'{tot_nr_new} new occurens_id(s) added to data and database'
                         )
                         )

        # self._post_event_result('nr_perfect_match', tot_nr_perfect_matches)
        # self._post_event_result('nr_added_valid_suggestions', tot_nr_added_valid_suggestions)
        # self._post_event_result('nr_new_ids', tot_nr_new)
        # self._post_event_result('valid_not_added', valid_not_added)
        return all_valid_matches

    def add_matching_to_data(self, df: pd.DataFrame, *matching_objects: DataTypeMatching):
        """Adds id to data from given matching objects"""
        for match_obj in matching_objects:
            boolean = df[self.temp_id_str_column] == match_obj.obj.temp_id_str
            df[boolean, self.id_column] = match_obj.match_uuid
            event.post_event('id_added_to_data_from_database',
                             dict(
                                 perfect_match_in_db=False,
                                 valid_match_in_db=match_obj.is_valid_match(),
                                 temp_id_str=match_obj.obj.temp_id_str,
                                 nr_places=len(np.where(boolean)[0]),
                                 forced=True,
                             )
                             )

    def _add_temp_id_str_column(self, df: pd.DataFrame) -> None:
        """Values in temp id column are concatenated from values of self.columns"""
        # if self.temp_id_str_column in df:
        #     return
        df[self.temp_id_str_column] = df[self.columns].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

    def _get_suggestion_in_db(self, obj: DataTypeDatabaseTable) -> DataTypeMatching | None:
        """Returns the best matches found in database."""
        # obj.add_all_cols_field()

        for i in range(len(obj.columns)):
            cols = obj.columns[:]
            cols.pop(i)
            result = self._get_db_match_for_columns(obj, cols)
            for res_obj in result:
                matching = self.matching_cls(obj, res_obj)
                if matching.is_valid_match():
                    return matching

    def old_get_suggestion_in_db(self, obj: DataTypeDatabaseTable, temp_id_str) -> list[DataTypeMatching]:
        """Returns the best matches found in database."""
        result = self._get_db_match_for_columns(obj, obj.columns)
        obj.temp_id_str = temp_id_str
        if result:
            return [self.matching_cls(obj, result[0])]
        all_matching = []
        for i in range(len(obj.columns)):
            cols = obj.columns[:]
            cols.pop(i)
            result = self._get_db_match_for_columns(obj, cols)
            all_matching.extend([self.matching_cls(obj, mobj) for mobj in result])
        return all_matching

    def _get_db_match_for_columns(self, obj: DataTypeDatabaseTable, cols: list[str]) -> list[
        Type[DataTypeDatabaseTable]]:
        with self.Session() as session:
            query = session.query(self._cls)
            for col in cols:
                value = getattr(obj, col)
                query = query.filter(getattr(self._cls, col) == value)
            result = query.all()
            return result

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

