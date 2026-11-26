# from nodc_occurrence_id.data_types.base import DataTypeDatabaseTable, DataTypeMatching


# @dataclass
# class GreySealDatabaseTable(DataTypeDatabaseTable):
#     data_type: str = field(default="greyseal", init=False)
#     reported_station_name: str | None = None
#     station_name: str | None = None
#     datetime_str: str | None = None
#     reported_position_str: str | None = (None,)
#     reported_scientific_name: str | None = (None,)
#
#     @property
#     def mandatory_columns(self) -> list[str]:
#         return [
#             "datetime_str",
#             "reported_station_name",
#             "station_name",
#             "reported_position_str",
#             "reported_scientific_name",
#         ]
#
#
# class GreySealDataTypeMatching(DataTypeMatching):
#     def is_valid_match(self) -> bool:
#         diff_columns = self.diff_columns
#         if len(diff_columns) == 2:
#             if diff_columns.get("reported_station_name") and diff_columns.get(
#                 "station_name"
#             ):
#                 if diff_columns.get("reported_station_name") == diff_columns.get(
#                     "station_name"
#                 ):
#                     return True
#             return False
#         return False
#
#
# class GreySealOccurrencesDatabase(OccurrencesDatabase):
#     data_type = "greyseal"
#     check_nr_diffs = 2
#     cls = GreySealDatabaseTable
#     matching_cls = GreySealDataTypeMatching
