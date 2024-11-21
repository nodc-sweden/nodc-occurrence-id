import sqlalchemy.orm as orm

from nodc_occurrence_id.data_types.base import Base, DataTypeDatabaseTable, OccurrencesDatabase, DataTypeMatching


class ZoobenthosDatabaseTable(Base, DataTypeDatabaseTable):
    __tablename__ = 'zoobenthos'

    sample_date: orm.Mapped[str]
    sample_time: orm.Mapped[str]
    reported_station_name: orm.Mapped[str]
    reported_scientific_name: orm.Mapped[str]
    species_flag_code: orm.Mapped[str]
    dev_stage_code: orm.Mapped[str]

    @property
    def mandatory_columns(self) -> list[str]:
        return [
            'sample_date',
            'sample_time',
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



