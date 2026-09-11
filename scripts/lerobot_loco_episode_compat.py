"""Compatibility patch for LeRobot episode-filter index construction."""

from __future__ import annotations

from functools import wraps


def install_episode_filter_compat() -> None:
    """Install the reversible DatasetReader episode-index compatibility patch."""
    from lerobot.datasets.dataset_reader import DatasetReader

    if getattr(DatasetReader, "_libero36_episode_compat_installed", False):
        return

    original = DatasetReader._build_index_mapping

    @wraps(original)
    def patched_build_index_mapping(self):
        if getattr(self, "episodes", None) is None:
            return original(self)

        dataset = self.hf_dataset
        original_format = dict(dataset.format)
        format_kwargs = dict(original_format.get("format_kwargs") or {})
        transform = format_kwargs.get("transform")
        if original_format.get("type") != "custom" or transform is None:
            return original(self)

        columns = original_format.get("columns")
        output_all_columns = bool(original_format.get("output_all_columns", False))
        dataset.reset_format()
        try:
            return original(self)
        finally:
            dataset.set_transform(
                transform,
                columns=columns,
                output_all_columns=output_all_columns,
            )

    DatasetReader._build_index_mapping = patched_build_index_mapping
    DatasetReader._libero36_episode_compat_installed = True
