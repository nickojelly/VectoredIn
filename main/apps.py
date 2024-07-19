from django.apps import AppConfig
from .utils import hnsw_intialize, initialize_openai_client, initialize_df, initialize_weaviate_client, initialize_data
from django.db.models.signals import post_migrate

class MainConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'main'

    def ready(self):
        from .models import QueryEmbedding, JobListing
        # Call the initialization function from utils
        initialize_df()
        hnsw_intialize()
        initialize_openai_client()
        initialize_weaviate_client()
        post_migrate.connect(initialize_data, sender=self)