from enum import StrEnum, auto


class Events(StrEnum):
    PROGRESS = auto()
    MISSING_MANDATORY_COLUMNS = auto()
    DATABASE_IS_UPDATED = auto()
    NR_PERFECT_MATCH = auto()
    NR_VALID_ADDED = auto()
    NR_VALID_NOT_ADDED = auto()
    NR_NEW = auto()


_subscribers = dict()
for e in Events:
    _subscribers[str(e)] = dict()


class EventNotFound(Exception):
    pass


def get_events() -> list[str]:
    return sorted(_subscribers)


def subscribe(event: str | Events, func, prio: int = 50) -> None:
    event = str(event)
    if event not in _subscribers:
        raise EventNotFound(event)
    _subscribers[event].setdefault(prio, [])
    if str(func) in [str(f) for f in _subscribers[event][prio]]:
        return
    _subscribers[event][prio].append(func)


def post_event(event: str, data: dict | None = None) -> None:
    if event not in _subscribers:
        raise EventNotFound(event)
    for prio in sorted(_subscribers[event]):
        for func in _subscribers[event][prio]:
            func(data or dict())
