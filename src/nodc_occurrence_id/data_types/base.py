import dataclasses
import datetime
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class DataTypeDatabaseTable:
    uuid: str | None = None
    create_time: str | None = None
    update_time: str | None = None
    all_cols: str | None = None

    def __post_init__(self):
        now = str(datetime.datetime.now())
        if not self.create_time:
            self.create_time = now
        if not self.update_time:
            self.update_time = now
        if not self.uuid:
            self.uuid = str(uuid.uuid4())

        self._fields = [f.name for f in dataclasses.fields(self) if f.name != "data_type"]
        self._columns = [
            col
            for col in self._fields
            if col not in ["uuid", "all_cols", "create_time", "update_time"]
        ]
        self._editable_columns = [
            col for col in self._fields if col not in ["create_time"]
        ]

    @property
    def fields(self) -> list[str]:
        return self._fields

    @property
    def columns(self) -> list[str]:
        return self._columns

    @property
    def editable_columns(self) -> list[str]:
        return self._editable_columns

    @property
    def fields_data(self) -> dict:
        return dict((col, getattr(self, col)) for col in self.fields)

    @property
    def data(self) -> dict:
        return dict((col, getattr(self, col)) for col in self.columns)

    @property
    def editable_data(self) -> dict:
        return dict((col, getattr(self, col)) for col in self.editable_columns)

    @property
    def nr_columns(self) -> int:
        return len(self.columns)

    @property
    def mandatory_columns(self) -> list[str]:
        return []

    @property
    def new_post_columns(self) -> list[str]:
        return []

    def set_all_cols_field(self) -> None:
        self.all_cols = "<>".join(self.data.values())


class DataTypeMatching(ABC):
    def __init__(
        self, obj: DataTypeDatabaseTable, match_obj: DataTypeDatabaseTable
    ) -> None:
        self.obj: DataTypeDatabaseTable = obj
        self.match_obj: DataTypeDatabaseTable = match_obj

    def __repr__(self):
        match_str = f"matching {self.match_uuid}" if self.is_valid_match() else "no match"
        diff_str_list = [
            f"{key}: {item['match_value']} (old) -> {item['value']} (new)"
            for key, item in self.diff_columns.items()
        ]
        diff_str = "  ;  ".join(diff_str_list)
        return f"Match object ({match_str}): {diff_str}"

    @property
    def columns(self) -> list[str]:
        return self.obj.columns[:]

    @property
    def nr_columns(self) -> int:
        return self.obj.nr_columns

    @property
    def match_uuid(self) -> str:
        return self.match_obj.uuid

    @property
    def editable_columns(self) -> list[str]:
        return self.obj.editable_columns

    def get_updated_data(self) -> dict | None:
        """Returns the updated data. uuid from match object. The rest from object.
        Also updates the update_time.
        Returns None if not valid"""
        if not self.is_valid_match():
            return
        self.obj.set_all_cols_field()
        data = self.obj.editable_data
        data["uuid"] = self.match_uuid
        data["update_time"] = str(datetime.datetime.now())
        return data

    @property
    def diff_columns(self) -> dict[str, dict[Any, Any]]:
        diffs = {}
        for col in self.columns:
            obj_val = getattr(self.obj, col)
            match_obj_val = getattr(self.match_obj, col)
            if obj_val == match_obj_val:
                continue
            diffs[col] = {}
            diffs[col]["value"] = obj_val
            diffs[col]["match_value"] = match_obj_val
            diffs[col]["match_uuid"] = self.match_obj.uuid
        return diffs

    @property
    def nr_diff_columns(self) -> int:
        return len(self.diff_columns)

    @property
    def percent_match(self):
        return round((self.nr_columns - self.nr_diff_columns) * 100 / self.nr_columns)

    @abstractmethod
    def is_valid_match(self) -> bool:
        """Set up rules that decides if the partial match is valid or not"""
        ...
