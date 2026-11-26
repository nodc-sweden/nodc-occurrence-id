import dataclasses
import datetime
import uuid
from abc import abstractmethod
from dataclasses import dataclass


@dataclass
class DataTypeDatabaseTable:
    occurrence_id: str | None = None
    create_time: str | None = None
    update_time: str | None = None
    all_cols: str | None = None

    def __post_init__(self):
        now = str(datetime.datetime.now())
        if not self.create_time:
            self.create_time = now
        if not self.update_time:
            self.update_time = now
        if not self.occurrence_id:
            self.occurrence_id = str(uuid.uuid4())

        self._fields = [f.name for f in dataclasses.fields(self) if f.name != "data_type"]
        self._columns = [
            col
            for col in self._fields
            if col not in ["occurrence_id", "all_cols", "create_time", "update_time"]
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

    @abstractmethod
    def is_valid_match(self) -> bool:
        """Set up rules that decides if the partial match is valid or not"""
        ...
