def serialize_order(order: dict[str, str]) -> str:
    return f"order:{order['id']}"
