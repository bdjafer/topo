from api.serializers import serialize_order


def load_recent() -> dict[str, str]:
    event = {"id": "recent"}
    serialize_order(event)
    return event
