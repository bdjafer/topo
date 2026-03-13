from api.serializers import serialize_order
from core.service import create_order, fetch_order
from data.audit import record_event
from data.store import load_order


def submit_order() -> str:
    order = create_order()
    record_event("submit")
    return serialize_order(order)


def get_order() -> str:
    order = fetch_order()
    fallback = load_order()
    return serialize_order(order if order["id"] else fallback)
