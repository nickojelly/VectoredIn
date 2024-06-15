from django.apps import AppConfig
from .utils import hnsw_intialize, initialize_openai_client, initialize_df

class MainConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'main'

    def ready(self):
        # Call the initialization function from utils
        hnsw_intialize()
        initialize_openai_client()
        initialize_df()