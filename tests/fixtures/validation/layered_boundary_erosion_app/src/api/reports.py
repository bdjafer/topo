from api.serializers import serialize_order
from data.audit import record_event
from data.store import load_order, save_order


def recent_order_report() -> str:
    order = load_order()
    record_event("report")
    save_order(order)
    return serialize_order(order)
