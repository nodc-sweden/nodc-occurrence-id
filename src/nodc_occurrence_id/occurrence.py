import datetime
import pathlib
import uuid
from typing import Any, Type

import polars as pl

from nodc_occurrence_id import event, utils
from nodc_occurrence_id.data_types.base import DataTypeDatabaseTable, DataTypeMatching


class OccurrencesDatabase:
    data_type: str = ""
    cls: Type[DataTypeDatabaseTable] = None
    matching_cls: Type[DataTypeMatching] = None
    check_nr_diffs = 1
    _name = "occurrence_id"  # This is the name of the id column in data

    def __init__(self, db_path: pathlib.Path | str) -> None:

        self._cls: Type[DataTypeDatabaseTable] = self.cls
        self._cls_obj: DataTypeDatabaseTable = self.cls()

        self.columns = self._cls_obj.columns[:]
        self.mandatory_columns = self._cls_obj.mandatory_columns[:]

        self._db_path = pathlib.Path(db_path)

        self.id_mapper: dict[str, str] = {}

        self._db_df: pl.DataFrame | None = None

        self._load_db()

    @property
    def db_path(self) -> pathlib.Path:
        return self._db_path

    def _load_db(self) -> None:
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
        return f"_{self._name}_str"

    def _get_table_obj(
        self, series: dict, include_uuid: bool = False, include_all_cols: bool = False
    ) -> DataTypeDatabaseTable | None:
        """Returns the Database table object if all mandatory columns are present"""
        ddict = self.filter_dict(
            series, include_uuid=include_uuid, include_all_cols=include_all_cols
        )
        missing_cols = [col for col in self.mandatory_columns if not ddict[col].strip()]
        if missing_cols:
            event.post_event(
                "missing_mandatory_columns",
                dict(
                    missing_columns=missing_cols, temp_id=series[self.temp_id_str_column]
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
            dict(
                (item, str(uuid.uuid4()))
                for item in set(self._db_df[self.temp_id_str_column])
            )
        )
        self._db_df = self._db_df.with_columns(
            pl.col(self.temp_id_str_column).replace_strict(self.id_mapper).alias("uuid")
        )
        now = str(datetime.datetime.now())
        self._db_df = self._db_df.with_columns(
            pl.lit(now).alias("create_time"),
            pl.lit(now).alias("update_time"),
            pl.col(self.temp_id_str_column).alias("all_cols"),
        )
        self._db_df = self._db_df[self._cls_obj.fields]
        return True

    def _handle_new_posts(self, data: pl.DataFrame) -> pl.DataFrame:
        # Not implemented
        return data

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

    def _add_objs_to_db(self, objs: list[DataTypeDatabaseTable]) -> None:
        """Adds all given objects to the database"""
        if not objs:
            return
        records = [obj.fields_data for obj in objs]
        if self._db_df.is_empty():
            df = pl.DataFrame(records)
        else:
            df = pl.concat([self._db_df, pl.DataFrame(records)])
        if df.height != df.unique(["all_cols"]).height:
            raise Exception(
                "Values in all_cols-column is not unique. Something whent wrong..."
            )
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
        for col, mapper in mapping.items():
            exps.append(
                pl.col("uuid").replace_strict(mapper, default=pl.col(col)).alias(col)
            )
        self._db_df = self._db_df.with_columns(exps)

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

    def _map_perfect_match(self, df: pl.DataFrame) -> tuple[pl.DataFrame, int]:
        perfect_match_df = df.join(
            self._db_df, left_on=self.temp_id_str_column, right_on="all_cols", how="inner"
        )
        no_perfect_match_df = df.join(
            self._db_df, left_on=self.temp_id_str_column, right_on="all_cols", how="anti"
        )
        id_perfect_match = dict(
            zip(perfect_match_df[self.temp_id_str_column], perfect_match_df["uuid"])
        )

        self.id_mapper.update(id_perfect_match)
        return (no_perfect_match_df, len(id_perfect_match))

    def _map_suggestions_in_db(
        self, no_perfect_match_df: pl.DataFrame, add_if_valid: bool = False
    ) -> dict[str, Any]:
        if not no_perfect_match_df.height:
            return {}
        self.no_perfect_match_df = no_perfect_match_df

        objs_to_add_to_db: list[DataTypeMatching] = []
        valid_matches_to_update_in_database: list[DataTypeMatching] = []
        valid_not_added: list[DataTypeMatching] = []
        self.cols = []

        for i in range(len(self.columns)):
            cols = self.columns[:]
            cols.pop(i)
            info = self._get_matches(no_perfect_match_df, cols)
            objs_to_add_to_db.extend(info["new"])
            if add_if_valid:
                valid_matches_to_update_in_database.extend(info["valid"])
            else:
                valid_not_added.extend(info["valid"])
        if self.check_nr_diffs == 2:
            no_perfect_match_df = no_perfect_match_df.filter(
                ~pl.col(self.temp_id_str_column).is_in(list(self.id_mapper))
            )
            if no_perfect_match_df.height:
                double_cols = self.columns + self.columns
                for n in range(len(self.columns)):
                    cols = double_cols[n + 1 :][: len(self.columns) - 2]
                    info = self._get_matches(no_perfect_match_df, cols)
                    objs_to_add_to_db.extend(info["new"])
                    if add_if_valid:
                        valid_matches_to_update_in_database.extend(info["valid"])
                    else:
                        valid_not_added.extend(info["valid"])

        return dict(
            objs_to_add_to_db=objs_to_add_to_db,
            tot_nr_new=len(objs_to_add_to_db),
            valid_matches_to_update_in_database=valid_matches_to_update_in_database,
            valid_not_added=valid_not_added,
        )

    def _get_matches(self, no_perfect_match_df: pl.DataFrame, cols: list[str]):
        temp_concat_col = "_temp_concat_col"
        db_df = self._db_df.with_columns(
            pl.concat_str(cols, separator="<>").alias(temp_concat_col)
        )
        df = no_perfect_match_df.with_columns(
            pl.concat_str(cols, separator="<>").alias(temp_concat_col)
        )
        match_df = df.join(db_df, on=temp_concat_col, how="inner")

        col_data = dict(
            cols=cols,
            match_df=match_df,
            obj=[],
            match_obj=[],
            mdf_data=[],
            valid_match=[],
        )
        if not match_df.height:
            self.cols.append(col_data)
            return dict(
                valid=[],
                new=[],
            )

        valid = list()
        new = list()

        right_cols = [col for col in match_df.columns if col.endswith("_right")]
        remove_cols_in_match = [col[:-6] for col in right_cols]
        right_cols_mapper = dict(zip(right_cols, remove_cols_in_match))

        for (temp_id_str,), mdf in match_df.group_by(temp_concat_col):
            obj = self._get_table_obj(
                df.filter(pl.col(temp_concat_col) == temp_id_str).to_dicts()[0]
            )

            mdf = mdf.drop(remove_cols_in_match)
            mdf = mdf.rename(right_cols_mapper)

            mdf_data = mdf.to_dicts()[0]

            self.mdf_data = mdf_data

            match_obj = self._get_table_obj(mdf_data, include_uuid=True)
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
                valid.append(valid_match)
            else:
                _id = str(uuid.uuid4())
                obj.uuid = _id
                new.append(obj)
                self.id_mapper[mdf_data[self.temp_id_str_column]] = _id
        self.cols.append(col_data)

        return dict(
            valid=valid,
            new=new,
        )

    def _old_map_suggestions_in_db(
        self, no_perfect_match_df: pl.DataFrame, add_if_valid: bool = False
    ) -> dict[str, Any]:
        if not no_perfect_match_df.height:
            return {}
        self.no_perfect_match_df = no_perfect_match_df
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
            match_df = df.join(db_df, on=temp_concat_col, how="inner")

            col_data = dict(
                cols=cols,
                match_df=match_df,
                obj=[],
                match_obj=[],
                mdf_data=[],
                valid_match=[],
            )
            if not match_df.height:
                self.cols.append(col_data)
                continue

            right_cols = [col for col in match_df.columns if col.endswith("_right")]
            remove_cols_in_match = [col[:-6] for col in right_cols]
            right_cols_mapper = dict(zip(right_cols, remove_cols_in_match))

            for (temp_id_str,), mdf in match_df.group_by(temp_concat_col):
                obj = self._get_table_obj(
                    df.filter(pl.col(temp_concat_col) == temp_id_str).to_dicts()[0]
                )

                mdf = mdf.drop(remove_cols_in_match)
                mdf = mdf.rename(right_cols_mapper)

                mdf_data = mdf.to_dicts()[0]

                self.mdf_data = mdf_data

                match_obj = self._get_table_obj(mdf_data, include_uuid=True)
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
                        self.id_mapper[mdf_data[self.temp_id_str_column]] = (
                            valid_match.match_uuid
                        )
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

    def add_uuid_to_data_and_database(
        self, df: pl.DataFrame, add_if_valid: bool = False
    ) -> pl.DataFrame:
        """Adds uuid to dataframe for all rows that have match in database.
        If not match in database a new id is created and added to dataframe and database.
        Option to also add if 'self.is_valid_match' if True
        (set flag add_if_valid=True)"""
        hashes_before = utils.get_database_hashes()
        if self.id_column not in df.columns:
            df = df.with_columns(pl.lit("").alias(self.id_column))

        # Preparing data
        df = self._add_temp_id_str_column(df=df)
        mask = pl.lit(True)
        for col in self.mandatory_columns:
            mask = mask & (pl.col(col) != "")
        missing_mandatory = len(df.filter(~mask))
        data = df.filter(mask)

        if self._handle_new_db(data):
            df = self._add_ids_to_df(df)
            nr_new = len(self._db_df)
            event.post_event(
                event.Events.NR_NEW,
                dict(
                    value=nr_new,
                    msg=f"{nr_new} new occurens_id(s) added to data and database",
                ),
            )
            self.save()
            hashes_after = utils.get_database_hashes()
            self._check_hashes(hashes_before, hashes_after)
            return df

        data = self._handle_new_posts(data)

        (no_perfect_match_df, tot_nr_perfect_matches) = self._map_perfect_match(data)
        suggestion_info = self._map_suggestions_in_db(
            no_perfect_match_df, add_if_valid=add_if_valid
        )

        df = self._add_ids_to_df(df)
        self._update_db_from_match_obj(
            suggestion_info.get("valid_matches_to_update_in_database")
        )
        self._add_objs_to_db(suggestion_info.get("objs_to_add_to_db"))

        if missing_mandatory:
            event.post_event(
                event.Events.MISSING_MANDATORY_COLUMNS,
                dict(
                    value=missing_mandatory,
                    msg=f"Mandatory columns missing in {missing_mandatory} rows",
                ),
            )

        if tot_nr_perfect_matches:
            event.post_event(
                event.Events.NR_PERFECT_MATCH,
                dict(
                    value=tot_nr_perfect_matches,
                    msg=f"Adding {tot_nr_perfect_matches} "
                    f"occurence_id(s) from perfect match in database",
                ),
            )

        if suggestion_info.get("valid_matches_to_update_in_database"):
            event.post_event(
                event.Events.NR_VALID_ADDED,
                dict(
                    value=suggestion_info.get("valid_matches_to_update_in_database"),
                    msg=f"Adding "
                    f"{len(suggestion_info.get('valid_matches_to_update_in_database'))}"
                    f" occurence_id(s) from VALID match in database. "
                    f"Database is updated!",
                ),
            )

        if suggestion_info.get("valid_not_added"):
            event.post_event(
                event.Events.NR_VALID_NOT_ADDED,
                dict(
                    value=suggestion_info.get("valid_not_added"),
                    msg=f"Found {len(suggestion_info.get('valid_not_added'))} "
                    f"VALID occurence_id match(es)in database but did not add! "
                    f"Set add_if_valid=True if you want to add and update them",
                ),
            )

        if suggestion_info.get("tot_nr_new"):
            event.post_event(
                event.Events.NR_NEW,
                dict(
                    value=suggestion_info.get("tot_nr_new"),
                    msg=f"{suggestion_info.get('tot_nr_new')} "
                    f"new occurens_id(s) added to data and database",
                ),
            )
        self.save()
        hashes_after = utils.get_database_hashes()
        self._check_hashes(hashes_before, hashes_after)
        return df

    def _check_hashes(self, before: dict, after: dict):
        files = dict(
            new_db=[],
            updated_db=[],
        )
        updated = False
        for name, h in after.items():
            if before.get(name) is None:
                files["new_db"].append(name)
                updated = True
            elif before.get(name) != h:
                files["updated_db"].append(name)
                updated = True
        if updated:
            event.post_event(event.Events.DATABASE_IS_UPDATED, files)

    def _post_event_progress(self, current: int, total: int) -> None:
        event.post_event(
            event.Events.PROGRESS,
            dict(
                total=total,
                current=current,
                title="Checking occurrence id",
            ),
        )

    def filter_dict(
        self, data: dict, include_uuid: bool = False, include_all_cols: bool = False
    ) -> dict:
        new_data = {}
        columns = self.columns[:]
        if include_uuid:
            columns.append("uuid")
        if include_all_cols:
            columns.append("all_cols")
        for col in columns:
            new_data[col] = data.get(col, "")
        return new_data
