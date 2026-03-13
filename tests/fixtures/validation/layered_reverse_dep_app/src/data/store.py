from api.serializers import serialize_order
from data.audit import record_event


def save_order(order: dict[str, str]) -> None:
    record_event(order["id"])


def load_order() -> dict[str, str]:
    order = {"id": "stored"}
    record_event("load")
    serialize_order(order)
    return order
