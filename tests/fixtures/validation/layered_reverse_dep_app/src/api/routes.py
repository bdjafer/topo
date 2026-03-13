from api.serializers import serialize_order
from core.service import create_order, fetch_order
from data.store import load_order


def submit_order() -> str:
    order = create_order()
    return serialize_order(order)


def get_order() -> str:
    return serialize_order(fetch_order())


def inspect_storage() -> str:
    return serialize_order(load_order())
