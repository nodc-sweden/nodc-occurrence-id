import sqlalchemy.orm as orm

from nodc_occurrence_id.data_types.base import Base, DataTypeDatabaseTable, OccurrencesDatabase, DataTypeMatching


class IfcbDatabaseTable(Base, DataTypeDatabaseTable):
    __tablename__ = 'ifcb'

    # reported_station_name: orm.Mapped[str]
    reported_station_name: orm.Mapped[str] = orm.mapped_column(index=True)
    reported_scientific_name: orm.Mapped[str]
    species_flag_code: orm.Mapped[str]

    @property
    def mandatory_columns(self) -> list[str]:
        return [
            'datetime_str',
            # 'sample_date',
            # 'sample_time',
            'reported_station_name',
            'reported_scientific_name'
        ]


class IfcbDataTypeMatching(DataTypeMatching):

    def is_valid_match(self) -> bool:
        if self.diff_columns.get('reported_scientific_name'):
            return False
        return True


class IfcbOccurrencesDatabase(OccurrencesDatabase):
    data_type = 'ifcb'
    cls = IfcbDatabaseTable
    matching_cls = IfcbDataTypeMatching



