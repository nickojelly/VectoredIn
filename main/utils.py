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

def initialize_openai_client():
    global openai_client  # Declare that you're modifying the global variable
    set_env()

    # Load the OpenAI API key from an environment variable
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    print(f"OpenAI API key loaded {openai_api_key}")

    # Initialize the OpenAI client

    # openai_client = openai.Client(api_key=openai_api_key)

    openai.api_key = openai_api_key
    openai_client = openai.OpenAI(api_key=openai_api_key)

    # return openai

listing_df: pd.DataFrame = None
embedding_df: pd.DataFrame = None
query_df: pd.DataFrame = None

def initialize_df() -> pd.DataFrame:
    global listing_df  # Declare that you're modifying the global variable

    df_path = os.path.join(settings.BASE_DIR, 'static', 'myapp', 'postings_10k.fth')

    listing_df = pd.read_feather(df_path)
    return listing_df 
    
weaviate_client = None  

def initialize_weaviate_client() :
    global weaviate_client  # Declare that you're modifying the global variable

    wdc_url = os.environ.get("WEAVIATE_URL")
    wdc_api_key = os.environ.get("WEAVIATE_KEY")
    openai_api_key = os.environ.get('OPENAI_API_KEY')

    print

    print(f"Itinializing Weaviate client with url {wdc_url} and api key {wdc_api_key}")
    
    weaviate_client = weaviate.connect_to_wcs(
        cluster_url=wdc_url,
        auth_credentials=weaviate.auth.AuthApiKey(wdc_api_key),
        headers={
        "X-OpenAI-Api-Key": openai_api_key  # Replace with your inference API key
    }
    )

    return weaviate_client

# Define a function to call the endpoint and obtain embeddings
def vectorize(openai_client, texts: List[str], rag_mode=False) -> List[List[float]]:

    prompt = f"""You are a job posting writer. Your task is to create a compelling and detailed job description for the role of {texts}. The job description should be around 200 words and should cover the following aspects:

1. A brief introduction to the role and its importance within the organization.
2. Key responsibilities and duties associated with the role.
3. Required qualifications, skills, and experience.
4. Desired personal attributes and qualities.
5. Information about the company culture, benefits, and growth opportunities.

Please write the job description in a professional and engaging tone, highlighting the exciting challenges and opportunities the role offers. Make sure to use persuasive language that would attract top talent to apply for the position.
"""

    if rag_mode:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are a job posting writer. Your task is to create a compelling and detailed job description for the role of {texts}. The job description should be around 200 words and should cover the following aspects:"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7,
        )

        embeddings =  openai_client.embeddings.create(
            input=[response], model="text-embedding-3-small"
        )
    else:
        embeddings = openai_client.embeddings.create(
            input=texts, model="text-embedding-3-small"
        )
        

    return embeddings.data[0].embedding