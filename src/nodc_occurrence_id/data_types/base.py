import dataclasses
from dataclasses import dataclass, field
import datetime
import pathlib
import uuid
from abc import ABC, abstractmethod
from typing import Any, Type
import numpy as np

import polars as pl

from nodc_occurrence_id import event


@dataclass
class DataTypeDatabaseTable:
    uuid: str | None = None
    create_time: str | None = None
    update_time: str | None = None
    all_cols: str | None = None

    def __post_init__(self):
        now = str(datetime.datetime.now())
        if not self.create_time:
            self.create_time = now
        if not self.update_time:
            self.update_time = now
        if not self.uuid:
            self.uuid = str(uuid.uuid4())

        self.__fields = [f.name for f in dataclasses.fields(self) if f.name != "data_type"]
        self.__columns = [col for col in self.__fields if col not in ['uuid',
                                                                      'all_cols',
                                                                      'create_time',
                                                                      'update_time']]
        self.__editable_columns = [col for col in self.__fields if col not in [
                                                                        'create_time']]

    @property
    def fields(self) -> list[str]:
        return self.__fields

    @property
    def columns(self) -> list[str]:
        return self.__columns

    @property
    def editable_columns(self) -> list[str]:
        return self.__editable_columns

    @property
    def fields_data(self) -> dict:
        return dict((col, getattr(self, col)) for col in self.fields)

    @property
    def data(self) -> dict:
        return dict((col, getattr(self, col)) for col in self.columns)

    @property
    def editable_data(self) -> dict:
        return dict((col, getattr(self, col)) for col in self.editable_columns)

    @property
    def nr_columns(self) -> int:
        return len(self.columns)

    @property
    def mandatory_columns(self) -> list[str]:
        return []

    @property
    def new_post_columns(self) -> list[str]:
        return []

    def set_all_cols_field(self) -> None:
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
    def editable_columns(self) -> list[str]:
        return self.obj.editable_columns

    def get_updated_data(self) -> dict | None:
        """Returns the updated data. uuid from match object. The rest from object.
        Also updates the update_time.
        Returns None if not valid"""
        if not self.is_valid_match():
            return
        self.obj.set_all_cols_field()
        data = self.obj.editable_data
        data["uuid"] = self.match_uuid
        data["update_time"] = str(datetime.datetime.now())
        return data

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

        self.db_path = pathlib.Path(db_path)

        self.id_mapper: dict[str, str] = {}

        self._db_df: pl.DataFrame | None = None
        if self.db_path.exists():
            self._db_df = pl.read_csv(
                self.db_path,
                encoding="utf8",
                separator="\t",
                infer_schema=False,
                missing_utf8_is_empty_string=True,
            )

    def save(self) -> None:
        self._db_df.write_csv(self.db_path, separator="\t")

    @property
    def id_column(self) -> str:
        return self._name

    @property
    def temp_id_str_column(self) -> str:
        return f'_{self._name}_str'

    def _get_table_obj(self, series: dict, include_uuid: bool = False, include_all_cols: bool = False) -> DataTypeDatabaseTable | None:
        """Returns the Database table object if all mandatory columns are present"""
        # print("--------------")
        # print(f"{series=}")
        # print(f"{series.get('uuid')=}")

        ddict = self.filter_dict(series, include_uuid=include_uuid, include_all_cols=include_all_cols)

        # print(f"{ddict=}")
        # print(f"{ddict.get('uuid')=}")
        # print()
        # print()
        # print()
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

    def _handle_new_db(self, data: pl.DataFrame) -> bool:
        if self._db_df is not None:
            return False
        self._db_df = data.unique(self.temp_id_str_column).sort(self.temp_id_str_column)
        self.id_mapper.update(
            dict((item, str(uuid.uuid4())) for item in set(self._db_df[self.temp_id_str_column]))
        )
        self._db_df = self._db_df.with_columns(pl.col(self.temp_id_str_column).replace_strict(self.id_mapper).alias("uuid"))
        now = str(datetime.datetime.now())
        self._db_df = self._db_df.with_columns(
            pl.lit(now).alias("create_time"),
            pl.lit(now).alias("update_time"),
            pl.col(self.temp_id_str_column).alias("all_cols"),
        )
        self._db_df = self._db_df[self._cls_obj.fields]
        return True

    def _handle_new_posts(self, data: pl.DataFrame) -> pl.DataFrame:
        return data
        # self._db_df = data.unique(self.temp_id_str_column).sort(self.temp_id_str_column)[self._cls_obj.fields]


    def _get_uuid_in_db_from_table_object(self, obj: DataTypeDatabaseTable) -> str | None:
        """
        Search for match in database and returns the corresponding uuid.
        A match is found if all values of self.columns match.
        Returns None if no match found.
        """
        ans = self._db_df.filter(pl.col("all_cols") == obj.all_cols)
        if ans.is_empty():
            return
        return ans["uuid"][0]

        # with self.Session() as session:
        #     res = session.query(self._cls).filter(self._cls.all_cols == obj.all_cols).first()
        #     if not res:
        #         return
        #     return res.uuid

    # def _get_suggestion_in_db(self, obj: DataTypeDatabaseTable) -> DataTypeMatching | None:
    #     """Returns the best matches found in database."""
    #     for i in range(len(obj.columns)):
    #         cols = obj.columns[:]
    #         cols.pop(i)
    #         result = self._get_db_match_for_columns(obj, cols)
    #         for res_obj in result:
    #             matching = self.matching_cls(obj, res_obj)
    #             # print(f"{matching.is_valid_match()=}")
    #             if matching.is_valid_match():
    #                 return matching

    # def _get_db_match_for_columns(self,
    #                               obj: DataTypeDatabaseTable,
    #                               # cols: list[str]) -> list[Type[DataTypeDatabaseTable]]:
    #                               cols: list[str]) -> list[DataTypeDatabaseTable]:
    #
    #     exps = []
    #     for col in cols:
    #         exps.append(pl.col(col) == getattr(obj, col))
    #     ans = self._db_df.filter(exps)
    #     if ans.is_empty():
    #         return []
    #     return [self._cls(**d) for d in ans.to_dicts()]

        # with self.Session() as session:
        #     query = session.query(self._cls)
        #     for col in cols:
        #         value = getattr(obj, col)
        #         query = query.filter(getattr(self._cls, col) == value)
        #     result = query.all()
        #     return result

    def _add_objs_to_db(self, objs: list[DataTypeDatabaseTable]) -> None:
        """Adds all given objects to the database"""
        # print()
        # print(f"_add_objs_to_db: {objs=}")
        # print()
        if not objs:
            return
        records = [obj.fields_data for obj in objs]
        if self._db_df.is_empty():
            df = pl.DataFrame(records)
        else:
            df = pl.concat([self._db_df, pl.DataFrame(records)])
        if df.height != df.unique(["all_cols"]).height:
            raise Exception("Values in all_cols-column is not unique. "
                            "Something whent wrong...")
        self._db_df = df

    def _update_db_from_match_obj(self, valid_matches: list[DataTypeMatching]) -> None:
        """Adds all valid objects to the database"""
        if not valid_matches:
            return
        mapping = dict((col, dict()) for col in valid_matches[0].editable_columns)
        for match in valid_matches:
            data = match.get_updated_data()
            if not data:
                continue
            _id = data.pop("uuid")
            for col, value in data.items():
                mapping[col][_id] = value
        self.mapping = mapping
        # Update dataframe
        exps = []
        # print(f"{mapping=}")
        for col, mapper in mapping.items():
            exps.append(pl.col("uuid").replace_strict(mapper, default=pl.col(col)).alias(col))
        self._db_df = self._db_df.with_columns(exps)



        # update_time = datetime.datetime.now(datetime.UTC)
        # with self.Session() as session:
        #     # updated = []
        #     for match in valid_matches:
        #         obj = session.query(self._cls).filter(self._cls.uuid == match.match_obj.uuid).first()
        #         for col, value in match.obj.data.items():
        #             if col in ['id', 'uuid', 'create_time']:
        #                 continue
        #             setattr(obj, col, value)
        #         obj.add_all_cols_field()
        #         obj.update_time = update_time
        #         # updated.append(obj)
        #     session.commit()

    def _add_temp_id_str_column(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.concat_str(self.columns, separator="<>").alias(self.temp_id_str_column)
        )

    def _add_ids_to_df(self, df: pl.DataFrame):
        return df.with_columns(
            pl.col(self.temp_id_str_column)
            .replace_strict(self.id_mapper, default="")
            .alias(self.id_column)
        )

    def _map_perfect_match(self,
                           df: pl.DataFrame
                           ) -> tuple[pl.DataFrame, pl.DataFrame, int]:
        perfect_match_df = df.join(self._db_df,
                             left_on=self.temp_id_str_column,
                             right_on="all_cols",
                             how="inner")
        no_perfect_match_df = df.join(self._db_df,
                              left_on=self.temp_id_str_column,
                              right_on="all_cols",
                              how="anti")
        id_perfect_match = dict(
            zip(perfect_match_df[self.temp_id_str_column], perfect_match_df["uuid"]))

        self.id_mapper.update(id_perfect_match)
        return (perfect_match_df,
                no_perfect_match_df,
                len(id_perfect_match))

    def _map_suggestions_in_db(self,
                               no_perfect_match_df: pl.DataFrame,
                               add_if_valid: bool = False) -> dict[str, Any]:
        if not no_perfect_match_df.height:
            return {}
        tot_nr_new = 0

        objs_to_add_to_db = []
        valid_matches_to_update_in_database: list[DataTypeMatching] = []
        valid_not_added: list[DataTypeMatching] = []
        self.cols = []

        temp_concat_col = "_temp_concat_col"
        for i in range(len(self.columns)):
            cols = self.columns[:]
            cols.pop(i)
            db_df = self._db_df.with_columns(
                pl.concat_str(cols, separator="<>").alias(temp_concat_col)
            )
            df = no_perfect_match_df.with_columns(
                pl.concat_str(cols, separator="<>").alias(temp_concat_col)
            )
            match_df = df.join(db_df,
                               on=temp_concat_col,
                               how="inner")
            print("match_df")
            print(f"{match_df.height=}")
            col_data = dict(
                cols=cols,
                match_df=match_df,
                obj=[],
                match_obj=[],
                mdf_data=[],
                valid_match=[])
            if not match_df.height:
                self.cols.append(col_data)
                continue
            # return match_df

            right_cols = [col for col in match_df.columns if
                          col.endswith("_right")]
            remove_cols_in_match = [col[:-6] for col in right_cols]
            right_cols_mapper = dict(zip(right_cols, remove_cols_in_match))

            for (temp_id_str,), mdf in match_df.group_by(temp_concat_col):

                obj = self._get_table_obj(df.filter(pl.col(temp_concat_col) ==
                                                    temp_id_str).to_dicts()[0])
                # print(f"{temp_concat_col=}")
                # print(f"{temp_id_str=}")
                # print(f"{db_df.filter(pl.col(temp_concat_col) == temp_id_str)=}")
                # match_df = db_df.filter(pl.col(temp_concat_col) == temp_id_str)
                # if not match_df:
                #     raise
                # if match_df.height:
                mdf = mdf.drop(remove_cols_in_match)
                mdf = mdf.rename(right_cols_mapper)

                mdf_data = mdf.to_dicts()[0]

                self.mdf_data = mdf_data

                match_obj = self._get_table_obj(mdf_data,
                                                include_uuid=True)
                self.obj = obj
                self.match_obj = match_obj

                obj.set_all_cols_field()
                match_obj.set_all_cols_field()
                valid_match = self.matching_cls(obj, match_obj)

                col_data["obj"].append(obj)
                col_data["match_obj"].append(match_obj)
                col_data["mdf_data"].append(mdf_data)
                col_data["valid_match"].append(valid_match)

                if valid_match.is_valid_match():
                    if add_if_valid:
                        valid_matches_to_update_in_database.append(valid_match)
                        self.id_mapper[mdf_data[self.temp_id_str_column]] = valid_match.match_uuid
                    else:
                        valid_not_added.append(valid_match)
                else:
                    _id = str(uuid.uuid4())
                    obj.uuid = _id
                    objs_to_add_to_db.append(obj)
                    self.id_mapper[mdf_data[self.temp_id_str_column]] = _id
                    tot_nr_new += 1
            self.cols.append(col_data)
        return dict(
            objs_to_add_to_db=objs_to_add_to_db,
            tot_nr_new=tot_nr_new,
            valid_matches_to_update_in_database=valid_matches_to_update_in_database,
            valid_not_added=valid_not_added,
        )

    def add_uuid_to_data_and_database(self, df: pl.DataFrame, add_if_valid: bool = False) -> pl.DataFrame:
        """Adds uuid to dataframe for all rows that have match in database.
        If not match in database a new id is created and added to dataframe and database.
        Option to also add if 'self.is_valid_match' if True (set flag add_if_valid=True)"""
        import time
        t0 = time.time()
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
        # return data
        # tot_occurrences = len(list(data.group_by(self.temp_id_str_column)))

        if self._handle_new_db(data):
            df = self._add_ids_to_df(df)
            self.save()
            return df

        data = self._handle_new_posts(data)

        (perfect_match_df,
         no_perfect_match_df,
         tot_nr_perfect_matches) = self._map_perfect_match(data)
        suggestion_info = self._map_suggestions_in_db(no_perfect_match_df,
                                                      add_if_valid=add_if_valid)

        self.perfect_match_df = perfect_match_df
        self.no_perfect_match_df = no_perfect_match_df
        self.tot_nr_perfect_matches = tot_nr_perfect_matches
        self.suggestion_info = suggestion_info

        print(f"{tot_nr_perfect_matches=}")
        print(f"{suggestion_info=}")

        # i = 0
        # for (temp_id_str, ), red_df in data.group_by(self.temp_id_str_column):
        #     series_dict = red_df[0].to_dicts()[0]
        #     obj = self._get_table_obj(series_dict)
        #     if not obj:
        #         continue
        #     obj.set_all_cols_field()
        #     if i and not i % 1000:
        #         self._post_event_progress(i, tot_occurrences)
        #     _id = self._get_uuid_in_db_from_table_object(obj)
        #     if _id:
        #         """ Perfect match in database. Add database UUID to dataframe"""
        #         id_mapper[temp_id_str] = _id
        #         tot_nr_perfect_matches += 1
        #     else:
        #         valid_match = self._get_suggestion_in_db(obj)
        #         if not valid_match:
        #             """No perfect match or valid suggestions in database"""
        #             _id = str(uuid.uuid4())
        #             obj.uuid = _id
        #             objs_to_add_to_db.append(obj)
        #             id_mapper[temp_id_str] = _id
        #             tot_nr_new += 1
        #         else:
        #             if add_if_valid:
        #                 valid_matches_to_update_in_database.append(valid_match)
        #                 id_mapper[temp_id_str] = valid_match.match_uuid
        #             else:
        #                 valid_not_added.append(valid_match)
        #     i += 1

        df = self._add_ids_to_df(df)
        self._update_db_from_match_obj(suggestion_info.get("valid_matches_to_update_in_database"))
        self._add_objs_to_db(suggestion_info.get("objs_to_add_to_db"))
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

        if suggestion_info.get("valid_matches_to_update_in_database"):
            event.post_event('result',
                             dict(
                                 name='valid_added',
                                 value=suggestion_info.get("valid_matches_to_update_in_database"),
                                 msg=f'Adding {len(suggestion_info.get("valid_matches_to_update_in_database"))} ")'
                                     f'occurence_id(s) from VALID match in database. '
                                     f'Database is updated!'
                             )
                             )

        if suggestion_info.get("valid_not_added"):
            event.post_event('result',
                             dict(
                                 name='valid_not_added',
                                 value=suggestion_info.get("valid_not_added"),
                                 msg=f'Found {len(suggestion_info.get("valid_not_added"))} '
                                     f'VALID occurence_id match(es)in database but did not add! '
                                     f'Set add_if_valid=True if you want to add and update them'
                             )
                             )

        if suggestion_info.get("tot_nr_new"):
            event.post_event('result',
                             dict(
                                 name='nr_new_ids',
                                 value=suggestion_info.get("tot_nr_new"),
                                 msg=f'{suggestion_info.get("tot_nr_new")} new occurens_id(s) added to data and database'
                             )
                             )
        self.save()
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
        columns = self.columns[:]
        if include_uuid:
            columns.append('uuid')
        if include_all_cols:
            columns.append('all_cols')
        for col in columns:
            new_data[col] = data.get(col, '')

            # from_col = col
            # if col == 'uuid':
            #     from_col = self.id_column
            # value = data.get(from_col, '')
            # new_data[col] = value
        return new_data
