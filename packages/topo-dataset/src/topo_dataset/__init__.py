"""topo-dataset: Dataset pipeline for R-GIN training."""

from topo_dataset.loader import load_graph
from topo_dataset.dataset import TopoDataset

__all__ = ["load_graph", "TopoDataset"]
