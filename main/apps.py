from django.apps import AppConfig
from .utils import hnsw_intialize, initialize_openai_client, initialize_df, initialize_weaviate_client

class MainConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'main'

    def ready(self):
        from .models import QueryEmbedding
        # Call the initialization function from utils
        hnsw_intialize()
        initialize_openai_client()
        initialize_weaviate_client()
        initialize_df()