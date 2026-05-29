from enum import StrEnum, auto


class Events(StrEnum):
    MISSING_MANDATORY_COLUMNS = auto()
    NO_ID_COLUMN_IN_DATA = auto()
    MISSING_ID_IN_DATA = auto()
    ID_ADDED_TO_DATABASE_FROM_DATA = auto()
    ID_ADDED_TO_DATA_FROM_DATABASE = auto()
    NEW_ID_ADDED_TO_DATA_AND_DATABASE = auto()
    SEVERAL_VALID_MATCHES_IN_DATABASE = auto()
    VALID_MATCH_IN_DATABASE = auto()
    PROGRESS = auto()
    RESULT = auto()
    DATABASE_IS_UPDATED = auto()


_subscribers = dict()
for e in Events:
    _subscribers[str(e)] = dict()


class EventNotFound(Exception):
    pass


def get_events() -> list[str]:
    return sorted(_subscribers)


def subscribe(event: str, func, prio: int = 50) -> None:
    if event not in _subscribers:
        raise EventNotFound(event)
    _subscribers[event].setdefault(prio, [])
    _subscribers[event][prio].append(func)


def post_event(event: str, data: dict | None = None) -> None:
    if event not in _subscribers:
        raise EventNotFound(event)
    for prio in sorted(_subscribers[event]):
        for func in _subscribers[event][prio]:
            func(data or dict())
