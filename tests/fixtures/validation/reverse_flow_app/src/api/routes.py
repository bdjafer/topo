from api.serializers import serialize_order
from data.store import load_recent


def recent_orders() -> str:
    return serialize_order(load_recent())
