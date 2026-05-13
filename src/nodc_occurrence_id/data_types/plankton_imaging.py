from dataclasses import dataclass, field

from nodc_occurrence_id.data_types.base import DataTypeDatabaseTable, DataTypeMatching
from nodc_occurrence_id.occurrence import OccurrencesDatabase


@dataclass
class PlanktonImagingDatabaseTable(DataTypeDatabaseTable):
    data_type: str = field(default="plankton_imaging", init=False)

    reported_station_name: str | None = None
    reported_scientific_name: str | None = None
    datetime_str: str | None = None
    species_flag_code: str | None = None

    @property
    def mandatory_columns(self) -> list[str]:
        return ["datetime_str", "reported_station_name", "reported_scientific_name"]

    @property
    def new_post_columns(self) -> list[str]:
        return [
            "datetime_str",
            "reported_station_name",
        ]


class PlanktonImagingDataTypeMatching(DataTypeMatching):
    def is_valid_match(self) -> bool:
        if self.diff_columns.get("reported_scientific_name"):
            return False
        return True


class PlanktonImagingOccurrencesDatabase(OccurrencesDatabase):
    data_type = "plankton_imaging"
    cls = PlanktonImagingDatabaseTable
    matching_cls = PlanktonImagingDataTypeMatching
