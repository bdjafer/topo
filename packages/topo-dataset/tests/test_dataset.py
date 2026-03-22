"""Tests for PyG Dataset class."""

import pytest

from topo_dataset.dataset import TopoDataset


class TestTopoDataset:
    def test_length(self, multi_repo_dir):
        examples_dir, split_file = multi_repo_dir
        dataset = TopoDataset(examples_dir, split_file)
        assert len(dataset) == 2

    def test_get_item(self, multi_repo_dir):
        examples_dir, split_file = multi_repo_dir
        dataset = TopoDataset(examples_dir, split_file)

        data = dataset[0]
        assert data["node"].x_semantic.shape[0] == 10
        assert data["node"].x_type.shape[0] == 10

    def test_iteration(self, multi_repo_dir):
        examples_dir, split_file = multi_repo_dir
        dataset = TopoDataset(examples_dir, split_file)

        graphs = list(dataset)
        assert len(graphs) == 2
        for g in graphs:
            assert g["node"].x_semantic.shape[1] == 768

    def test_empty_split_file(self, multi_repo_dir, tmp_path):
        examples_dir, _ = multi_repo_dir
        empty_split = tmp_path / "empty.txt"
        empty_split.write_text("")
        dataset = TopoDataset(examples_dir, empty_split)
        assert len(dataset) == 0

    def test_skips_blank_lines(self, multi_repo_dir, tmp_path):
        examples_dir, _ = multi_repo_dir
        split = tmp_path / "with_blanks.txt"
        split.write_text("repo_a\n\nrepo_b\n\n")
        dataset = TopoDataset(examples_dir, split)
        assert len(dataset) == 2
