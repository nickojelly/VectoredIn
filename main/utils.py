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
# from .models import JobPosting, QueryEmbedding, JobListing
from tqdm import tqdm
import json
hnsw_obj: hnsw_python.HNSW = None

def hnsw_intialize() -> hnsw_python.HNSW:
    global hnsw_obj  # Declare that you're modifying the global variable

    hnsw_path = os.path.join(settings.BASE_DIR, 'static', 'myapp', 'hnsw_save_dict_V2.pkl')
    df_path = os.path.join(settings.BASE_DIR, 'static', 'myapp', 'postings_w_embeddings_v2.fth')
    with open(hnsw_path, 'rb') as f:
        hnsw_save_dict = pickle.load(f)
    df = listing_df

    # Initialize the hnsw_obj instance
    hnsw_obj = hnsw_python.HNSW('cosine', df, m=20, m0=40, ef=100)
    hnsw_obj._graphs = hnsw_save_dict['graphs']
    hnsw_obj._enter_point = hnsw_save_dict['entrypoint']
    hnsw_obj.data = hnsw_save_dict['data']

    print(f"hnsw_obj initialized")

    return hnsw_obj

# Define a module-level variable to store the openai_client instance
openai_client = None

def initialize_openai_client():
    global openai_client  # Declare that you're modifying the global variable
    set_env()

    # Load the OpenAI API key from an environment variable
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    print(f"OpenAI API key loaded {openai_api_key}")

    openai.api_key = openai_api_key
    openai_client = openai.OpenAI(api_key=openai_api_key)

    # return openai

listing_df: pd.DataFrame = None
embedding_df: pd.DataFrame = None
query_df: pd.DataFrame = None

def initialize_df() -> pd.DataFrame:
    global listing_df  # Declare that you're modifying the global variable

    df_path = os.path.join(settings.BASE_DIR, 'static', 'myapp', 'data_wv_uuid.fth')

    listing_df = pd.read_feather(df_path)
    # listing_df = None
    print(f"listing_df initialized")

    return listing_df 
    
weaviate_client = None  

from django.db import transaction

def initialize_data(**kwargs):
    from .models import JobListing
    df_path = os.path.join(settings.BASE_DIR, 'static', 'myapp', 'postings_w_embeddings_v2.fth')
    listing_df = pd.read_feather(df_path)
    print(f"Starting to create JobListing objects")
    JobListing.objects.all().delete()
    
    job_listings = []
    for _, row in tqdm(listing_df.iterrows()):
        print()
        job_listings.append(JobListing(
            job_id=int(row['job_id']),
            wv_uuid=row['wv_uuid'],
            company_name=row['company_name'],
            company_id=int(row['company_id']),
            title=row['title'],
            text=row['text'],
            annotations=json.dumps(row['annotations'].tolist()),
            vector=json.dumps(row['vector'].tolist())
        ))
    
    with transaction.atomic():
        JobListing.objects.bulk_create(job_listings, ignore_conflicts=True)
    
    print(f"Created {len(job_listings)} JobListing objects")


def initialize_weaviate_client() :
    global weaviate_client  # Declare that you're modifying the global variable

    wdc_url = os.environ.get("WCD_URL")
    wdc_api_key = os.environ.get("WEAVIATE_KEY")
    openai_api_key = os.environ.get('OPENAI_API_KEY')


    print(f"Itinializing Weaviate client with url {wdc_url}")
    print(wdc_url)
    print('45.79.238.197')
    
    weaviate_client = weaviate.connect_to_local(
        host=wdc_url,
        auth_credentials=weaviate.auth.AuthApiKey(wdc_api_key),
        headers={
        "X-OpenAI-Api-Key": openai_api_key  # Replace with your inference API key
    }
    )

    print(weaviate_client)

    return weaviate_client

