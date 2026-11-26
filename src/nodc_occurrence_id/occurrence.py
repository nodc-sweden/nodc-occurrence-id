import datetime
import pathlib
import uuid
from dataclasses import dataclass

import polars as pl
from nodc_config import Config

from nodc_occurrence_id import event, utils


@dataclass
class DatabaseMatching:
    count: int
    msg: str
    id_mapper: dict[str, str]


class OccurrencesDatabase:
    data_type: str = ""
    columns: tuple[str, ...] = None
    mandatory_columns: tuple[str, ...] = None
    _db_match_suffix = "_db"
    _df_match_suffix = "_df"

    def __init__(
        self,
        nodc_conf: Config,
        db_path: pathlib.Path | str,
        sort_db: bool = True,
        **kwargs,
    ) -> None:

        self._nodc_conf = nodc_conf
        self._sort_db = sort_db

        self._db_path = pathlib.Path(db_path)

        self.id_mapper: dict[str, str] = {}
        self.db_mappers: dict[str, dict[str, str]] = {}

        self._db_df: pl.DataFrame = pl.DataFrame()

        self._load_db()
        self._add_missing_columns_in_database()
        self._update_all_cols_column_in_database()

    @property
    def db(self) -> pl.DataFrame:
        return self._db_df

    @property
    def db_path(self) -> pathlib.Path:
        return self._db_path

    @property
    def id_column(self) -> str:
        return "occurrence_id"

    @property
    def all_cols_column(self) -> str:
        return "all_cols"

    @property
    def all_columns(self) -> list[str]:
        return [*self.columns, "occurrence_id", "create_time", "update_time", "all_cols"]

    @property
    def match_columns(self) -> list[str]:
        return self.columns

    @property
    def db_match_columns(self) -> list[str]:
        return [f"{col}{self._db_match_suffix}" for col in self.match_columns]

    @property
    def sort_column(self) -> str:
        return "datetime_str"

    def _get_db_match_column(self, col: str) -> str:
        return f"{col}{self._db_match_suffix}"

    def _get_df_match_column(self, col: str) -> str:
        return f"{col}{self._df_match_suffix}"

    def _load_db(self) -> None:
        if not self.db_path.exists():
            return
        self._db_df = pl.read_csv(
            self.db_path,
            encoding="utf8",
            separator="\t",
            infer_schema=False,
            missing_utf8_is_empty_string=True,
        )
        self._db_df = self._db_df.fill_null("").fill_nan("")

    def _add_missing_columns_in_database(self) -> None:
        """
        Adds columns that are not in database.
        This is so that we can extend the number of columns we use for checking unique
        occurrences.
        """
        self._db_df = self._db_df.with_columns(
            [
                pl.lit("").alias(col)
                for col in self.columns
                if col not in self._db_df.columns
            ]
        )

    def _update_all_cols_column_in_database(self):
        """
        We ude this because we need to be able to change perfect
        match-columns (property self.match_columns in child classes since the matching
        critera may change over time
        """
        self._db_df = self._add_all_cols_column(self._db_df)

    def save(self) -> None:
        if self._sort_db:
            self._db_df = self._db_df.sort(self.sort_column)
        self._db_df.write_csv(self.db_path, separator="\t")

    def _prepare_data(self, df: pl.DataFrame) -> pl.DataFrame:
        # Use this method in child classes if you need to prepare data before checking
        # dataframe. This should not filter data but keep all data and add extra
        # columns to bi able to match described columns.
        return df

    def _filter_data(self, df: pl.DataFrame) -> pl.DataFrame:
        # Use this method in child classes if you need to filter tha data that should
        # be used for checking matches.
        return df

    def _get_valid_matches(self, no_perfect_match_df: pl.DataFrame):
        raise NotImplementedError(
            "Method _get_valid_matches must be implemented by child class"
        )

    def add_occurrence_id_to_data_and_database(
        self,
        df: pl.DataFrame,
        add_if_valid: bool = False,
        force_new_occurrences: list[str] | None = None,
    ) -> pl.DataFrame:
        """Adds occurrence_id to dataframe for all rows that have match in database.
        If not match in database a new id is created and added to dataframe and database.
        Option to also add if 'self.is_valid_match' if True
        (set flag add_if_valid=True)"""

        df = self._prepare_data(df)

        hashes_before = utils.get_database_hashes(self._nodc_conf)
        reported_id_columns = f"reported_{self.id_column}"
        if self.id_column in df.columns:
            df = df.with_columns(pl.col(self.id_column).alias(reported_id_columns))
        else:
            df = df.with_columns(
                pl.lit("").alias(self.id_column), pl.lit("").alias(reported_id_columns)
            )

        # Preparing data
        df = self._add_all_cols_column(df=df)
        mask = pl.lit(True)
        for col in self.mandatory_columns:
            mask = mask & (pl.col(col) != "")
        missing_df = df.filter(~mask)
        missing_mandatory = dict()
        if missing_df.height:
            for col in self.mandatory_columns:
                col_df = missing_df.filter(pl.col(col) == "")
                if not col_df.height:
                    continue
                missing_mandatory[col] = len(col_df)

        if missing_mandatory:
            for col, count in missing_mandatory.items():
                event.post_event(
                    event.Events.MISSING_MANDATORY_COLUMNS,
                    dict(
                        value=missing_mandatory,
                        msg=f"Mandatory column '{col}' missing in {count} rows",
                    ),
                )

        data = df.filter(mask)
        data = self._filter_data(data)

        if self._handle_new_db(data):
            df = self._add_ids_to_df(df)
            nr_new = len(self._db_df)
            event.post_event(
                event.Events.NR_NEW,
                dict(
                    value=nr_new,
                    msg=f"{nr_new} new occurrence_id(s) added to data and database",
                ),
            )
            self.save()
            hashes_after = utils.get_database_hashes(self._nodc_conf)
            self._check_hashes(hashes_before, hashes_after)
            return df

        mapper = self._handle_new_posts(data)  # Not implemented
        self.id_mapper.update(mapper)
        data = self._add_ids_to_df(data, mapper=mapper)

        # Add ids for perfect match
        mapper = self._map_perfect_match(data)
        if mapper:
            self.id_mapper.update(mapper)
            data = self._add_ids_to_df(data, mapper=mapper)

        # No need to continue if all ids are set
        if self._all_ids_found(df):
            return df

        # Add suggestions from database and update (if valid)
        mapper = self._map_suggestions(
            data, add_if_valid=add_if_valid, force_new_occurrences=force_new_occurrences
        )

        if mapper:
            save_path = self._save_valid_match_file(list(mapper.keys()))
            event.post_event(
                event.Events.INFO,
                dict(
                    msg=f"Valid matches stored in file: {save_path}", data=sorted(mapper)
                ),
            )
            if add_if_valid:
                self.id_mapper.update(mapper)
                data = self._add_ids_to_df(data, mapper=mapper)
            else:
                data = data.filter(~pl.col(self.all_cols_column).is_in(mapper.keys()))

        # Add new if no match
        mapper = self._add_no_match_as_new(
            data, force_new_occurrences=force_new_occurrences
        )
        self.id_mapper.update(mapper)
        data = self._add_ids_to_df(data, mapper=mapper)

        # Finally map all to original data
        df = self._add_ids_to_df(df, mapper=self.id_mapper)

        # Save and check
        self.save()
        hashes_after = utils.get_database_hashes(self._nodc_conf)
        self._check_hashes(hashes_before, hashes_after)
        return df

    def _add_ids_to_df(self, df: pl.DataFrame, mapper: dict[str, str] | None = None):
        """Adds matching occurrence_id to the given dataframe"""
        if mapper is None:
            mapper = self.id_mapper
        return df.with_columns(
            pl.col(self.all_cols_column)
            .replace_strict(mapper, default=pl.col(self.id_column))
            .alias(self.id_column)
        )

    def _update_database_info(self):
        """Adds updates info in database"""
        all_ids = set()
        for col, mapper in self.db_mappers.items():
            self._db_df = self._db_df.with_columns(
                pl.col(self.id_column)
                .replace_strict(mapper, default=pl.col(col))
                .alias(col)
            )
            all_ids.update(set(mapper.keys()))

        # Update time
        now = str(datetime.datetime.now())
        mapper = dict((_id, now) for _id in all_ids)
        self._db_df = self._db_df.with_columns(
            pl.col(self.id_column)
            .replace_strict(mapper, default=pl.col("update_time"))
            .alias("update_time")
        )

        # Update all_cols column
        self._db_df = self._add_all_cols_column(self._db_df)

    def _map_perfect_match(self, df: pl.DataFrame) -> dict[str, str]:
        perfect_match_df = df.join(
            self._db_df,
            on=self.all_cols_column,
            how="inner",
            suffix=self._db_match_suffix,
        )

        id_column = self._get_db_match_column(self.id_column)
        if id_column not in perfect_match_df.columns:
            id_column = self.id_column

        mapper = dict(
            zip(perfect_match_df[self.all_cols_column], perfect_match_df[id_column])
        )
        if len(perfect_match_df):
            event.post_event(
                event.Events.NR_PERFECT_MATCH,
                dict(
                    value=len(perfect_match_df),
                    msg=f"{len(perfect_match_df)} occurrence_id(s) with perfect match "
                    f"in database",
                ),
            )
        # No need to update anything in database
        return mapper

    def _map_suggestions(
        self,
        df: pl.DataFrame,
        add_if_valid: bool = False,
        force_new_occurrences: list[str] | None = None,
    ) -> dict[str, str]:

        force_new_occurrences = force_new_occurrences or []
        # Filter data that does not have occurrence_id
        no_perfect_match_df = df.filter(pl.col(self.id_column) == "")
        if not no_perfect_match_df.height:
            return {}

        # Check in child classes if there are any valid non-perfect matches
        valid_matches: list[DatabaseMatching] = self._get_valid_matches(
            no_perfect_match_df
        )
        # TODO: Check if duplicates in matching mappers
        # TODO: Check given list for new IDS

        if not valid_matches:
            return {}

        # Update and log
        id_mapper = dict()
        for match in valid_matches:
            mapper = match.id_mapper.copy()
            for all_cols in force_new_occurrences:
                mapper.pop(all_cols, None)

            if not len(mapper):
                continue

            # Add to id_mapper
            id_mapper.update(match.id_mapper)

            # Add to data
            filter_df = df.filter(
                pl.col(self.all_cols_column).is_in(match.id_mapper.keys())
            )
            enriched_df = filter_df.with_columns(
                pl.col(self.all_cols_column)
                .replace_strict(mapper, default="")
                .alias(self.id_column)
            )

            match_df = enriched_df.join(
                self.db, on=self.id_column, how="inner", suffix=self._db_match_suffix
            )
            for col in self.columns:
                self.db_mappers.setdefault(col, dict())
                self.db_mappers[col].update(
                    dict(zip(match_df[self.id_column], match_df[col]))
                )

            # Log
            if not add_if_valid:
                event.post_event(
                    event.Events.NR_VALID_NOT_ADDED,
                    dict(value=len(mapper), msg=match.msg + f" ({len(mapper)} places)"),
                )
            else:
                event.post_event(
                    event.Events.NR_VALID_ADDED,
                    dict(
                        value=match.count,
                        msg=match.msg,
                    ),
                )

        if add_if_valid:
            self._update_database_info()

        return id_mapper

    def _add_no_match_as_new(
        self, df: pl.DataFrame, force_new_occurrences: list[str] | None = None
    ) -> dict[str, str]:
        df_no_id = df.unique(self.all_cols_column).filter(pl.col(self.id_column) == "")
        df_no_match = df_no_id.filter(
            ~pl.col(self.all_cols_column).is_in(self.id_mapper.keys())
        )

        # Add new ids
        df_with_id = df_no_match.with_columns(
            pl.Series(
                self.id_column, [str(uuid.uuid4()) for _ in range(len(df_no_match))]
            )
        )

        # Add time stamps and filter columns
        now = str(datetime.datetime.now())
        df_with_id = df_with_id.with_columns(
            pl.lit(now).alias("create_time"),
            pl.lit(now).alias("update_time"),
        )
        df_with_id = df_with_id[self.all_columns]

        # Create mapper for adding ids to original dataframe
        mapper = dict(zip(df_no_match[self.all_cols_column], df_no_match[self.id_column]))

        # Add ids to database
        self._db_df = pl.concat([self._db_df, df_with_id], how="diagonal")

        if nr_new := len(df_with_id):
            if force_new_occurrences:
                nr_forced = len(
                    [item for item in force_new_occurrences if item in mapper]
                )
                if nr_forced:
                    event.post_event(
                        event.Events.NR_NEW,
                        dict(
                            value=nr_new,
                            msg=f"{nr_new} new occurrence_id(s) forced by user added to "
                            f"data and database",
                        ),
                    )
                if nr_new := (nr_new - nr_forced):
                    event.post_event(
                        event.Events.NR_NEW,
                        dict(
                            value=nr_new,
                            msg=f"{nr_new} new occurrence_id(s) with no match added to "
                            f"data and database",
                        ),
                    )

        return mapper

    def _handle_new_db(self, data: pl.DataFrame) -> bool:
        if self._db_df is not None:
            return False
        self._db_df = data.unique(self.all_cols_column).sort(self.all_cols_column)
        self.id_mapper.update(
            dict(
                (item, str(uuid.uuid4()))
                for item in set(self._db_df[self.all_cols_column])
            )
        )
        self._db_df = self._db_df.with_columns(
            pl.col(self.all_cols_column)
            .replace_strict(self.id_mapper)
            .alias(self.id_column)
        )
        now = str(datetime.datetime.now())
        self._db_df = self._db_df.with_columns(
            pl.lit(now).alias("create_time"),
            pl.lit(now).alias("update_time"),
        )
        self._db_df = self._db_df[self.columns]
        return True

    def _handle_new_posts(self, data: pl.DataFrame) -> dict[str, str]:
        # Not implemented
        return {}

    def _add_all_cols_column(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.concat_str(self.match_columns, separator="<>").alias(self.all_cols_column)
        )

    def _all_ids_found(self, df: pl.DataFrame) -> bool:
        if df.filter(pl.col(self.id_column) == "").height:
            return False
        return True

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

    def _save_valid_match_file(self, all_cols_list: list[str]) -> pathlib.Path:
        path = pathlib.Path.home() / "occurrence_id" / "valid_matches.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fid:
            fid.write("\n".join(sorted(all_cols_list)))
        return path
