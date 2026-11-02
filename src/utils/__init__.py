"""Utility functions for BIGCLAM project."""

__all__ = ["prepare_gse96058_data", "prepare_metabric", "prepare_tcga_brca_data"]


def __getattr__(name):
    """Load dataset helpers on demand so ``python -m`` does not double-import them."""
    if name in __all__:
        from src.utils import prepare_datasets

        return getattr(prepare_datasets, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
