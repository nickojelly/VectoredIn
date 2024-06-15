import pickle
import os
from .hnsw import hnsw_python
from django.conf import settings
import pandas as pd
import openai
import weaviate
from typing import List
from .keys import set_env
# Define a module-level variable to store the hnsw_obj instance
hnsw_obj: hnsw_python.HNSW = None

def hnsw_intialize() -> hnsw_python.HNSW:
    global hnsw_obj  # Declare that you're modifying the global variable

    hnsw_path = os.path.join(settings.BASE_DIR, 'static', 'myapp', 'hnsw_save_dict.pkl')
    df_path = os.path.join(settings.BASE_DIR, 'static', 'myapp', 'postings_10k.fth')
    with open(hnsw_path, 'rb') as f:
        hnsw_save_dict = pickle.load(f)
    df = pd.read_feather(df_path)

    # Initialize the hnsw_obj instance
    hnsw_obj = hnsw_python.HNSW('cosine', df, m=20, m0=40, ef=100)
    hnsw_obj._graphs = hnsw_save_dict['graphs']
    hnsw_obj._enter_point = hnsw_save_dict['entrypoint']
    hnsw_obj.data = hnsw_save_dict['data']

    return hnsw_obj

# Define a module-level variable to store the openai_client instance
openai_client = None

def initialize_openai_client() -> openai.Client:
    global openai_client  # Declare that you're modifying the global variable
    set_env()

    # Load the OpenAI API key from an environment variable
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    print(f"OpenAI API key loaded {openai_api_key}")

    # Initialize the OpenAI client
    openai_client = openai.Client(api_key=openai_api_key)

    print(f"OpenAI API key loaded {openai_client}")

    return openai_client

listing_df: pd.DataFrame = None

def initialize_df() -> pd.DataFrame:
    global listing_df  # Declare that you're modifying the global variable

    df_path = os.path.join(settings.BASE_DIR, 'static', 'myapp', 'postings_10k.fth')

    listing_df = pd.read_feather(df_path)
    return listing_df 
    
weaviate_client = None  

def initialize_weaviate_client() -> openai.Client:
    global weaviate_client  # Declare that you're modifying the global variable
    
    weaviate_client = weaviate.connect_to_wcs(
    cluster_url=os.getenv("WCD_URL"),
    auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WCD_API_KEY")),
    headers={
        "X-OpenAI-Api-Key": os.environ["OPENAI_APIKEY"]  # Replace with your inference API key
        }
    )

    return weaviate_client

# Define a function to call the endpoint and obtain embeddings
def vectorize(openai_client, texts: List[str]) -> List[List[float]]:

    response = openai_client.embeddings.create(
        input=texts, model="text-embedding-3-small"
    )

    return response.data[0].embedding