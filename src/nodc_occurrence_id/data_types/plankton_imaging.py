import polars as pl

from nodc_occurrence_id.occurrence import DatabaseMatching, OccurrencesDatabase


class PlanktonImagingOccurrencesDatabase(OccurrencesDatabase):
    data_type = "plankton_imaging"
    columns = (
        "sample_id",
        "reported_scientific_name",
        "biovolume",  # Added for future use 2026-11
    )
    mandatory_columns = ("sample_id", "reported_scientific_name", "biovolume")

    @property
    def match_columns(self) -> list[str]:
        # TODO: Add biovolume at the right time
        return [
            "sample_id",
            "reported_scientific_name",
        ]

    @property
    def sort_column(self) -> str:
        return "sample_id"

    def _prepare_data(self, df: pl.DataFrame) -> pl.DataFrame:
        copy_columns = [("COPY_VARIABLE.Biovolume concentration.mm3/l", "biovolume")]
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

    def _get_valid_matches(
        self, no_perfect_match_df: pl.DataFrame
    ) -> list[DatabaseMatching]:
        return []
