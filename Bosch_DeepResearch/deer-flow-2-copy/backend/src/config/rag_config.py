import logging
import os
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RAGFilterConfig(BaseModel):
    """Configuration for filtering RAG resources (databases)"""

    enabled: bool = Field(default=False, description="Enable RAG resource filtering")
    whitelist: list[str] = Field(
        default_factory=list,
        description="List of dataset names to include (only these will be used). If empty, all datasets are used.",
    )
    blacklist: list[str] = Field(
        default_factory=list,
        description="List of dataset names to exclude. Whitelist takes precedence over blacklist.",
    )

    def should_include(self, dataset_name: str) -> bool:
        """Check if a dataset should be included based on filter configuration.
        
        Args:
            dataset_name: Name of the dataset to check
            
        Returns:
            True if the dataset should be included, False otherwise
        """
        if not self.enabled:
            return True

        # Whitelist takes precedence
        if self.whitelist:
            return dataset_name in self.whitelist

        # If no whitelist, use blacklist
        return dataset_name not in self.blacklist


class RAGConfig(BaseModel):
    """Complete RAG configuration"""

    provider: str | None = Field(default=None, description="RAG provider type (ragflow, etc)")
    filter: RAGFilterConfig = Field(default_factory=RAGFilterConfig, description="Resource filtering configuration")

    @classmethod
    def from_dict(cls, data: dict | None) -> Self:
        """Load RAG config from dictionary.
        
        Args:
            data: Dictionary containing RAG configuration
            
        Returns:
            RAGConfig instance
        """
        if not data:
            return cls()

        rag_config = {
            "provider": data.get("provider"),
        }

        # Load filter config if present
        if "filter" in data:
            filter_data = data.get("filter", {})
            rag_config["filter"] = RAGFilterConfig(
                enabled=filter_data.get("enabled", False),
                whitelist=filter_data.get("whitelist", []),
                blacklist=filter_data.get("blacklist", []),
            )

        return cls(**rag_config)


_rag_config: RAGConfig | None = None


def load_rag_config(config_data: dict | None) -> None:
    """Load and store RAG configuration from application config.
    
    Args:
        config_data: Dictionary containing the 'rag' key from config.yaml
    """
    global _rag_config
    _rag_config = RAGConfig.from_dict(config_data)
    if _rag_config.filter.enabled:
        logger.info(f"RAG filtering enabled: whitelist={_rag_config.filter.whitelist}, blacklist={_rag_config.filter.blacklist}")


def get_rag_config() -> RAGConfig:
    """Get the current RAG configuration.
    
    Returns:
        RAGConfig instance (returns default if not loaded yet)
    """
    global _rag_config
    if _rag_config is None:
        _rag_config = RAGConfig()
    return _rag_config
