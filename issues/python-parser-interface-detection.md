# Python Parser: Detect interfaces (ABCs, Protocols) as `NodeKind.INTERFACE`

## Problem

The Python parser classifies ALL classes as `kind=class`, including abstract base classes (`abc.ABC`), protocols (`typing.Protocol`), and classes with `@abstractmethod`. This means the R-GIN model's 4th node type (`interface`, index 3) is never populated for Python codebases.

The Rust parser already handles this correctly — it maps `trait` → `interface`. The Python parser needs the same treatment.

## Current Behavior

```python
# This gets kind=class (wrong)
class Shape(ABC):
    @abstractmethod
    def area(self): ...

# This also gets kind=class (wrong)
class Serializable(Protocol):
    def serialize(self) -> bytes: ...
```

In `python.py:_extract_class()`, every class is unconditionally assigned `NodeKind.CLASS`:
```python
graph.add_node(Node(id=class_id, kind=NodeKind.CLASS, ...))
```

## Expected Behavior

Classify as `NodeKind.INTERFACE` when any of:
1. Class inherits from `abc.ABC`
2. Class uses `metaclass=abc.ABCMeta`
3. Class inherits from `typing.Protocol` or `typing_extensions.Protocol`
4. Class has **all** methods decorated with `@abstractmethod`

## Why This Matters

- The R-GIN model has a dedicated embedding for `interface` nodes (index 3 in `NODE_TYPE_VOCAB`). Without interface detection, this embedding is never trained on Python data.
- Interfaces occupy structurally distinct positions — they're dependency hubs that many classes implement but rarely import from each other. The model can't learn this pattern if they're lumped with regular classes.
- Flask has `~46 classes` — some (like view base classes) are abstract contracts, not concrete types.

## Implementation Guide

### Where to change

`packages/topo-parser-python/src/topo_parser_python/python.py` — the `_extract_class()` function.

### Detection logic

```python
def _is_interface(node: ast.ClassDef) -> bool:
    """Check if a class is an abstract contract (ABC, Protocol, or pure abstract)."""
    # 1. Check base classes for ABC or Protocol
    for base in node.bases:
        name = _base_name(base)
        if name in {"ABC", "abc.ABC", "Protocol", "typing.Protocol",
                     "typing_extensions.Protocol"}:
            return True

    # 2. Check for metaclass=ABCMeta
    for kw in node.keywords:
        if kw.arg == "metaclass":
            name = _base_name(kw.value)
            if name in {"ABCMeta", "abc.ABCMeta"}:
                return True

    # 3. Check if ALL methods are abstract
    methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if methods and all(_has_abstractmethod(m) for m in methods):
        return True

    return False

def _base_name(node: ast.expr) -> str:
    """Extract name from a base class AST node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_base_name(node.value)}.{node.attr}"
    return ""

def _has_abstractmethod(func: ast.FunctionDef) -> bool:
    return any(
        _base_name(d) in {"abstractmethod", "abc.abstractmethod",
                           "abstractproperty", "abc.abstractproperty"}
        for d in func.decorator_list
    )
```

Then in `_extract_class()`:
```python
kind = NodeKind.INTERFACE if _is_interface(node) else NodeKind.CLASS
graph.add_node(Node(id=class_id, kind=kind, ...))
```

### What already works

- `NodeKind.INTERFACE = "interface"` is already defined in `graph.py`
- The Rust `NODE_TYPE_VOCAB` already has `"interface"` at index 3
- The `node_type_index()` function in `types.rs` already maps `"interface"` → 3
- The R-GIN embedding table already has a slot for index 3

### Testing

- Parse Flask: `flask.views.View` and `flask.views.MethodView` should become `interface`
- Parse a repo with `typing.Protocol` usage
- Parse `attrs` — `attrs.Attribute` is a concrete class, should stay `class`
- Verify the graph.json output has `"kind": "interface"` for detected ABCs

### Edge cases

- A class that inherits from ABC but has no abstract methods — still `interface` (it's declaring itself as abstract)
- A class that inherits from a Protocol subclass — only direct Protocol parents, not transitive (would require import resolution)
- `@abstractmethod` on some but not all methods — `class` (it's a partial implementation, closer to a mixin)

## Scope

- Python parser only. Rust parser already works.
- No changes needed to `types.rs`, `node_type_index`, or the R-GIN model.
- After fixing, re-run `topo export-features` on all examples to regenerate `features.npz` with correct type indices.
