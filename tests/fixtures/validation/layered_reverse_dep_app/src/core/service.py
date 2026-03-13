from core.rules import normalize_order
from data.store import load_order, save_order


def create_order() -> dict[str, str]:
    order = normalize_order({"id": "new"})
    save_order(order)
    return order


def fetch_order() -> dict[str, str]:
    order = load_order()
    return normalize_order(order)
