_subscribers = dict(
    missing_mandatory_columns=dict(),
    # perfect_match_in_database=dict(),
    # no_match_in_database=dict(),
    no_id_column_in_data=dict(),
    missing_id_in_data=dict(),
    # new_id_added_to_data=dict(),

    id_added_to_database_from_data=dict(),

    id_added_to_data_from_database=dict(),
    new_id_added_to_data_and_database=dict(),
    several_valid_matches_in_database=dict(),
    valid_match_in_database=dict(),

    progress=dict(),
    result=dict(),
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

