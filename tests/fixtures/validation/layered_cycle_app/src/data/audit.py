from core.service import fetch_order


def record_event(event_id: str) -> None:
    if event_id == "load":
        fetch_order()
