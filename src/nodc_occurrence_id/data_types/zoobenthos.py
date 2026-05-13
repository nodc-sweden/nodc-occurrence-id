from dataclasses import dataclass, field

from nodc_occurrence_id.data_types.base import DataTypeDatabaseTable, OccurrencesDatabase, DataTypeMatching


@dataclass
class ZoobenthosDatabaseTable(DataTypeDatabaseTable):
    data_type: str = field(default='zoobenthos', init=False)
    reported_station_name: str | None = None
    reported_scientific_name: str | None = None
    datetime_str: str | None = None
    species_flag_code: str | None = None
    dev_stage_code: str | None = None

    @property
    def mandatory_columns(self) -> list[str]:
        return [
            'datetime_str',
            # 'sample_date',
            # 'sample_time',
            'reported_station_name',
            'reported_scientific_name'
        ]


class ZoobenthosDataTypeMatching(DataTypeMatching):

    def is_valid_match(self) -> bool:
        if self.diff_columns.get('reported_scientific_name'):
            return False
        return True


class ZoobenthosOccurrencesDatabase(OccurrencesDatabase):
    data_type = 'zoobenthos'
    cls = ZoobenthosDatabaseTable
    matching_cls = ZoobenthosDataTypeMatching



