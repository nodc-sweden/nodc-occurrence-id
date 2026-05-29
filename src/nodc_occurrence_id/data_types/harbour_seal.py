from dataclasses import dataclass, field

from nodc_occurrence_id.data_types.base import DataTypeDatabaseTable, DataTypeMatching
from nodc_occurrence_id.occurrence import OccurrencesDatabase


@dataclass
class HarbourSealDatabaseTable(DataTypeDatabaseTable):
    data_type: str = field(default="harbourseal", init=False)
    reported_station_name: str | None = None
    station_name: str | None = None
    datetime_str: str | None = None
    reported_position_str: str | None = (None,)
    reported_scientific_name: str | None = (None,)

    @property
    def mandatory_columns(self) -> list[str]:
        return ["datetime_str", "reported_station_name", "station_name"]


class HarbourSealDataTypeMatching(DataTypeMatching):
    def is_valid_match(self) -> bool:
        return False


class HarbourSealOccurrencesDatabase(OccurrencesDatabase):
    data_type = "harbourseal"
    cls = HarbourSealDatabaseTable
    matching_cls = HarbourSealDataTypeMatching
