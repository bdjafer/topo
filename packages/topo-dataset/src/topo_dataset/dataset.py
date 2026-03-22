"""PyG Dataset class for topo training data.

Reads split files (train.txt, val.txt, test.txt) and loads
preprocessed graphs from the examples directory.
"""

from pathlib import Path

from torch_geometric.data import Dataset, HeteroData

from topo_dataset.loader import load_graph


class TopoDataset(Dataset):
    """PyG Dataset backed by preprocessed NPZ files.

    Args:
        examples_dir: Path to the examples/ directory containing repo subdirs.
        split_file: Path to a split file (one repo name per line).
        transform: Optional PyG transform to apply per-graph.
    """

    def __init__(
        self,
        examples_dir: Path,
        split_file: Path,
        transform=None,
    ):
        self.repo_names = [
            line.strip()
            for line in split_file.read_text().strip().split("\n")
            if line.strip()
        ]
        self.examples_dir = Path(examples_dir)
        super().__init__(root=None, transform=transform)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def download(self):
        pass

    def process(self):
        pass

    def len(self) -> int:
        return len(self.repo_names)

    def get(self, idx: int) -> HeteroData:
        name = self.repo_names[idx]
        return load_graph(self.examples_dir / name)
