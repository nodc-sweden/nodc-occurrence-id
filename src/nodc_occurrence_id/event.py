_subscribers = dict(
    missing_mandatory_columns=dict(),
    perfect_match_in_database=dict(),
    no_match_in_database=dict(),
)


class EventNotFound(Exception):
    pass


def get_events() -> list[str]:
    return sorted(_subscribers)


def subscribe(event: str, func, prio: int = 50) -> None:
    if event not in _subscribers:
        raise EventNotFound(event)
    _subscribers[event].setdefault(prio, [])
    _subscribers[event][prio].append(func)


def post_event(event: str, data: dict = None) -> None:
    if event not in _subscribers:
        raise EventNotFound(event)
    for prio in sorted(_subscribers[event]):
        for func in _subscribers[event][prio]:
            func(data or dict())

