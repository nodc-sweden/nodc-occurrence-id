import polars as pl

from nodc_occurrence_id.occurrence import DatabaseMatching, OccurrencesDatabase


class HarbourSealOccurrencesDatabase(OccurrencesDatabase):
    data_type = "harbourseal"
    columns = (
        "datetime_str",
        "reported_station_name",
        "station_name",
        "reported_position_str",
        "reported_scientific_name",
        "# counted",  # NY
    )
    mandatory_columns = (
        "datetime_str",
        "reported_station_name",
        "station_name",
        "reported_position_str",
        "reported_scientific_name",
        "COPY_VARIABLE.# counted.ind",  # This is translated to # counted in
        # self.prepare_data
    )

    def _prepare_data(self, df: pl.DataFrame) -> pl.DataFrame:
        copy_columns = [("COPY_VARIABLE.# counted.ind", "# counted")]
        exp = []
        for fr, to in copy_columns:
            if to in df.columns:
                continue
            if fr in df.columns:
                exp.append(pl.col(fr).alias(to))
            else:
                exp.append(pl.lit("").alias(to))
        if exp:
            df = df.with_columns(exp)
        return df

    def _filter_data(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(~pl.col("# counted").is_in(["", "0"]))

    def _get_valid_matches(
        self, no_perfect_match_df: pl.DataFrame
    ) -> list[DatabaseMatching]:
        matches = [
            self._get_valid_match_updated_reported_station_name(no_perfect_match_df),
            self._get_valid_match_updated_translated_station_name(no_perfect_match_df),
            self._get_valid_match_updated_count(no_perfect_match_df),
            self._get_valid_match_both_station_names_updated(no_perfect_match_df),
        ]
        return [match for match in matches if match is not None]

    def _get_valid_match_updated_reported_station_name(
        self, no_perfect_match_df: pl.DataFrame
    ) -> DatabaseMatching | None:
        temp_all_cols = [
            "datetime_str",
            "station_name",
            "reported_position_str",
            "reported_scientific_name",
            "# counted",
        ]
        return self._get_one_col_diff_match(
            no_perfect_match_df, temp_all_cols, "reported_station_name"
        )

    def _get_valid_match_updated_translated_station_name(
        self, no_perfect_match_df: pl.DataFrame
    ) -> DatabaseMatching | None:
        temp_all_cols = [
            "datetime_str",
            "reported_station_name",
            "reported_position_str",
            "reported_scientific_name",
            "# counted",
        ]
        return self._get_one_col_diff_match(
            no_perfect_match_df, temp_all_cols, "station_name"
        )

    def _get_valid_match_updated_count(
        self, no_perfect_match_df: pl.DataFrame
    ) -> DatabaseMatching | None:
        temp_all_cols = [
            "datetime_str",
            "reported_station_name",
            "station_name",
            "reported_position_str",
            "reported_scientific_name",
        ]
        return self._get_one_col_diff_match(
            no_perfect_match_df, temp_all_cols, "# counted"
        )

    def _get_valid_match_both_station_names_updated(
        self, no_perfect_match_df: pl.DataFrame
    ) -> DatabaseMatching | None:

        df = no_perfect_match_df.with_columns(
            pl.col("reported_station_name") == pl.col("station_name")
        )
        db_df = self._db_df.with_columns(
            pl.col("reported_station_name") == pl.col("station_name")
        )
        temp_all_cols = [
            "datetime_str",
            "reported_position_str",
            "reported_scientific_name",
            "# counted",
        ]
        temp_concat_col = "temp_all_cols"
        df = df.with_columns(
            pl.concat_str(temp_all_cols, separator="<>").alias(temp_concat_col)
        ).unique(temp_concat_col)

        db_df = db_df.with_columns(
            pl.concat_str(temp_all_cols, separator="<>").alias(temp_concat_col)
        ).unique(temp_concat_col)

        match_df = df.join(
            db_df, on=temp_concat_col, how="inner", suffix=self._db_match_suffix
        )

        if nr_new := len(match_df):
            mapper = dict(
                zip(
                    match_df["all_cols"],
                    match_df[self._get_db_match_column(self.id_column)],
                )
            )

            return DatabaseMatching(
                count=nr_new,
                msg="Valid match: 'reported_station_name' and 'station_name' is the "
                "same but has changed",
                id_mapper=mapper,
            )
        return None

    def _get_one_col_diff_match(
        self,
        no_perfect_match_df: pl.DataFrame,
        matching_columns: list[str],
        missing_col: str,
    ) -> DatabaseMatching | None:
        temp_concat_col = "temp_all_cols"
        df = no_perfect_match_df.with_columns(
            pl.concat_str(matching_columns, separator="<>").alias(temp_concat_col)
        ).unique(temp_concat_col)

        db_df = self._db_df.with_columns(
            pl.concat_str(matching_columns, separator="<>").alias(temp_concat_col)
        ).unique(temp_concat_col)

        match_df = df.join(
            db_df, on=temp_concat_col, how="inner", suffix=self._db_match_suffix
        )

        if nr_new := len(match_df):
            mapper = dict(
                zip(
                    match_df["all_cols"],
                    match_df[self._get_db_match_column(self.id_column)],
                )
            )

            return DatabaseMatching(
                count=nr_new,
                msg=f"Valid match: '{missing_col}' changed",
                id_mapper=mapper,
            )
        return None
