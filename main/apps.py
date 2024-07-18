from django.apps import AppConfig
from django.utils.functional import lazy
import pandas as pd
class MainConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'main'

    def ready(self):
        from .utils import (
            hnsw_initialize, 
            initialize_openai_client, 
            initialize_df, 
            initialize_weaviate_client
        )
        
        # Use lazy loading for initializations
        self.get_df = lazy(initialize_df, pd.DataFrame)
        self.get_hnsw = lazy(hnsw_initialize, object)
        self.get_openai_client = lazy(initialize_openai_client, object)
        self.get_weaviate_client = lazy(initialize_weaviate_client, object)

