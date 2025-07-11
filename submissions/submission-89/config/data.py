"""
Data configuration for GMM transformer framework.

This module defines the configuration settings for data generation and loading.
"""

from typing import Dict, Any, Optional, Union
from pydantic import Field, model_validator

from .base import ConfigBase


class DataConfig(ConfigBase):
    """Configuration for data generation and loading."""
    
    dim: int = Field(
        default=2,
        ge=1,
        description="Dimension of the data points"
    )
    
    cluster_params: Union[str, int, Dict[str, Any]] = Field(
        default="moderate",
        description="Configuration for number of clusters (string preset, int for fixed count, or dict)"
    )
    
    snr_db_params: Union[str, float, Dict[str, Any]] = Field(
        default="moderate",
        description="Configuration for signal-to-noise ratio (string preset, float for fixed SNR, or dict)"
    )
    
    alpha_dirichlet: float = Field(
        default=1.0,
        gt=0,
        description="Concentration parameter for Dirichlet distribution"
    )
    
    sample_count_distribution: Union[str, int, Dict[str, Any]] = Field(
        default="medium",
        description="Configuration for number of samples per GMM (string preset, int for fixed count, or dict)"
    )
    
    vary_clusters_in_batch: bool = Field(
        default=False,
        description="DEPRECATED: This option is no longer supported and will always be False. Varying cluster counts within a batch caused tensor dimension incompatibility issues."
    )
    
    vary_snr_in_batch: bool = Field(
        default=True,
        description="Whether to vary SNR within a batch"
    )
    
    uniform_sample_count: bool = Field(
        default=True,
        description="Whether to use the same sample count for all GMMs in batch"
    )
    
    random_seed: Optional[int] = Field(
        default=42,
        description="Random seed for reproducibility"
    )

    @model_validator(mode='after')
    def resolve_parameter_presets(self) -> "DataConfig":
        """
        Resolve string presets to their explicit parameter dictionaries.
        This happens after validation to ensure all values are already validated.
        """
        try:
            from data.utils.distribution_utils import resolve_preset
            
            # Resolve string presets to their dictionary equivalents
            if isinstance(self.cluster_params, str):
                self.cluster_params = resolve_preset(self.cluster_params, 'cluster_params')
                
            if isinstance(self.snr_db_params, str):
                self.snr_db_params = resolve_preset(self.snr_db_params, 'snr_db_params')
                
            if isinstance(self.sample_count_distribution, str):
                self.sample_count_distribution = resolve_preset(
                    self.sample_count_distribution, 'sample_count_distribution'
                )
        except ImportError:
            # If the module is not found, we'll leave the string presets as is
            # They will be resolved later when creating the generator
            pass
            
        return self
    
    def create_gmm_generator(self):
        """
        Create a parameter generator and data generator based on the configuration.
        
        Returns:
            DataGenerator instance
        """
        from data.core.parameter_generator import ParameterGenerator
        from data.core.data_generator import DataGenerator
        
        # Create the parameter generator
        param_gen = ParameterGenerator(
            dim=self.dim,
            cluster_config=self.cluster_params,
            snr_config=self.snr_db_params,
            sample_count_config=self.sample_count_distribution,
            alpha_dirichlet=self.alpha_dirichlet,
            seed=self.random_seed
        )
        
        # Create the data generator
        return DataGenerator(
            param_generator=param_gen,
            vary_clusters_in_batch=self.vary_clusters_in_batch,
            vary_control_in_batch=self.vary_snr_in_batch,
            seed=self.random_seed
        )