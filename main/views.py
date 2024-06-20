from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot
from django.templatetags.static import static
from django.conf import settings
import os
from .forms import PlotForm
from django.http import JsonResponse
import logging
from plotly.utils import PlotlyJSONEncoder
import json
from .utils import hnsw_obj,openai_client,listing_df,weaviate_client
from .generative import gen_utils
from django.views.decorators.csrf import csrf_exempt
import numpy as np
from .models import JobPosting, QueryEmbedding  # Import the JobPosting model
logger = logging.getLogger(__name__)

def about(request):
    return render(request, 'about.html')

def contact(request):
    return render(request, 'contact.html')

def homepage(request):
    return HttpResponse('Hello, World!')

def update_plot_data(request, x_text, y_text, z_text,k,n):
    rag = request.GET.get('rag', 'false') == 'true'
    return get_plot_data(request, x_text, y_text, z_text, k,n,update = True,rag=rag)

def get_plot_data(request, x_text, y_text, z_text,k=10,n=5, update = False,rag=False):

    plot_div, plot_data, plot_layout, corr = generate_plot(x_text, y_text, z_text,k,n, update = update,rag=rag)

    # Convert plot_data to a JSON-serializable format
    plot_data_json = json.dumps(plot_data, cls=PlotlyJSONEncoder)

    response_data = {
        'data': plot_data_json, 
        'layout': plot_layout,
        'corr': corr
    }

    return JsonResponse(response_data)


def vectorize_query_db(text, rag=False):
    # Check if the query already has been vectorized in the local database
    query_embedding = QueryEmbedding.objects.filter(query=text).first()

    if not query_embedding:
        # If the query embedding doesn't exist, create a new one
        vector = gen_utils.vectorize(openai_client,weaviate_client, text)
        vector_rag = gen_utils.vectorize(openai_client,weaviate_client, text, rag=True) if rag else None
        query_embedding = QueryEmbedding(query=text, query_embedding=json.dumps(vector), rag_query_embedding=json.dumps(vector_rag) if rag else None)
        query_embedding.save()
    else:
        # If the query embedding exists, retrieve the vectors
        vector = np.array(json.loads(query_embedding.query_embedding))
        if rag:
            if not query_embedding.rag_query_embedding:
                vector_rag = gen_utils.vectorize(openai_client,weaviate_client, text, rag=True)
                query_embedding.rag_query_embedding = json.dumps(vector_rag)
                query_embedding.save()
            vector = np.array(json.loads(query_embedding.rag_query_embedding))

    return vector

def update_data(x_text, y_text, z_text, k=5,n=10,rag=False):

    try:
        x_vector  = vectorize_query_db(x_text,rag)
        y_vector  = vectorize_query_db(y_text,rag)
        z_vector  = vectorize_query_db(z_text,rag)
    except Exception as e:
        print(f"Error in vectorizing query: {e}")
        x_vector  = vectorize_query_db(x_text,False)
        y_vector  = vectorize_query_db(y_text,False)
        z_vector  = vectorize_query_db(z_text,False)

    # Rest of the code remains the same
    x_list = hnsw_obj.serach_along_axis(x_vector, k,n=n)
    y_list = hnsw_obj.serach_along_axis(y_vector, k,n=n)
    z_list = hnsw_obj.serach_along_axis(z_vector, k,n=n)
    full_list = x_list | y_list | z_list

    distance_list = []
    for uuid, array in full_list.items():
        x_dist = hnsw_obj.distance(x_vector, array)
        y_dist = hnsw_obj.distance(y_vector, array)
        z_dist = hnsw_obj.distance(z_vector, array)
        distance_list.append((uuid, x_dist, y_dist, z_dist))

    distance_df = pd.DataFrame(data=distance_list, columns=['uuid', 'x_dist', 'y_dist', 'z_dist'])
    distance_df = distance_df.merge(listing_df, left_on='uuid', right_on='wv_uuid')

    return distance_df

def generate_plot(x_text, y_text, z_text,k=10,n=5, update=False,rag=False):
    # QueryEmbedding.objects.all().delete()
    #Process the data:
    if update:
        distance_df = update_data(x_text, y_text, z_text,k,n,rag)
    else: 
        distance_df_path = os.path.join(settings.BASE_DIR, 'static', 'myapp', 'distance_df.fth')
        distance_df = pd.read_feather(distance_df_path)
    # distance_df = distance_df.sample(n=10)
    x = distance_df['x_dist'].values
    y = distance_df['y_dist'].values
    z = distance_df['z_dist'].values
    custom_data = distance_df.wv_uuid.values
    global_min = min(x.min(), y.min(), z.min())
    global_max = max(x.max(), y.max(), z.max())

    xy_cor = np.corrcoef(x, y)[0,1]+1
    xz_cor = np.corrcoef(x, z)[0,1]+1
    yz_cor = np.corrcoef(y, z)[0,1]+1

    trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=8,
            color=x+y+z,
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title='Semantic Distance')
        ),
        text=distance_df['title']+' | '+distance_df['company_name'],
        hoverinfo='text',
        name='Semantic Distance',
        customdata=custom_data
    )

    layout = dict(
        scene=dict(
            aspectmode='cube',
            xaxis=dict(
                range=[global_min, global_max],
                title=x_text,
                backgroundcolor="rgb(200, 230, 225)",  # Muted color for x-axis
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
            ),
            yaxis=dict(
                range=[global_min, global_max],
                title=y_text,
                backgroundcolor="rgb(220, 230, 240)",  # Muted color for y-axis
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
            ),
            zaxis=dict(
                range=[global_min, global_max],
                title=z_text,
                backgroundcolor="rgb(230, 240, 220)",  # Muted color for z-axis
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
            ),
        ),
        hovermode='closest',
        width=800,  # Adjust the width as needed
        height=600,  # Adjust the height as needed
        margin=dict(r=10, l=10, b=10, t=10),
    )


    

    plot_div = plot({'data': [trace], 'layout': layout}, output_type='div', config={'responsive': True})

    return plot_div, [trace], layout, [xy_cor, xz_cor, yz_cor]


def test_hnsw(request):
    result = hnsw_obj.serach_along_axis(q=hnsw_obj.data[0],k=5)
    # print(f"Result of search = {result}")


def plot_view(request):
    initial_x_text = 'Machine Learning Engineer'
    initial_y_text = 'Data Scientist'
    initial_z_text = 'Accountant'

    form = PlotForm()

    plot_div, plot_data, plot_layout, corr = generate_plot(initial_x_text, initial_y_text, initial_z_text)

    context = {
        'form': form,
        'initial_x_text': initial_x_text,
        'initial_y_text': initial_y_text,
        'initial_z_text': initial_z_text,
        'plot_div': plot_div,
        'plot_data': json.dumps(plot_data, cls=PlotlyJSONEncoder),
        'plot_layout': plot_layout,
        'corr': corr,
    }

    return render(request, "base.html", context)

@csrf_exempt
def get_point_summary(request):
    if request.method == 'POST':
        point_data = json.loads(request.body)
        uuid = point_data['customData']
        x = point_data['x']
        y = point_data['y']
        z = point_data['z']

        job_listing = listing_df.query('wv_uuid == @uuid')

        if not job_listing.empty:
            summary_data = {
                'title': job_listing.iloc[0]['title'],
                'company': job_listing.iloc[0]['company_name'],
                'location': job_listing.iloc[0]['location'],
                'description': job_listing.iloc[0]['description'],
                'x': x,
                'y': y,
                'z': z
            }
            return JsonResponse({'summary': summary_data})
        else:
            return JsonResponse({'summary': None})
        
@csrf_exempt
def generate_plot_summary(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        # uuid = data.get('uuid')
        x_text = data.get('x_text', '')
        y_text = data.get('y_text', '')
        z_text = data.get('z_text', '')

        distance_df_path = os.path.join(settings.BASE_DIR, 'static', 'myapp', 'distance_df.fth')
        distance_df = pd.read_feather(distance_df_path)

        x_max = distance_df['x_dist'].max()
        x_min = distance_df['x_dist'].min()
        y_max = distance_df['y_dist'].max()
        y_min = distance_df['y_dist'].min()
        z_max = distance_df['z_dist'].max()
        z_min = distance_df['z_dist'].min()

        min_x_uuid = distance_df.query('x_dist == @x_min').wv_uuid.values[0]
        min_y_uuid = distance_df.query('y_dist == @y_min').wv_uuid.values[0]
        min_z_uuid = distance_df.query('z_dist == @z_min').wv_uuid.values[0]

        uuids = [min_x_uuid, min_y_uuid, min_z_uuid]

        text = (x_text, y_text, z_text)


        plot_summary = gen_utils.generate_plot_summary(uuids,text, client = weaviate_client, openai_client=openai_client)
        return JsonResponse({'summary': plot_summary})

    return JsonResponse({'summary': None})
    
@csrf_exempt
def get_alignment_summary(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        uuid = data.get('uuid')
        x_text = data.get('x_text', '')
        y_text = data.get('y_text', '')
        z_text = data.get('z_text', '')

        distance_df_path = os.path.join(settings.BASE_DIR, 'static', 'myapp', 'distance_df.fth')
        distance_df = pd.read_feather(distance_df_path)

        x_max = distance_df['x_dist'].max()
        x_min = distance_df['x_dist'].min()
        y_max = distance_df['y_dist'].max()
        y_min = distance_df['y_dist'].min()
        z_max = distance_df['z_dist'].max()
        z_min = distance_df['z_dist'].min()

        dist_ranges = {
            'x_range': [x_min, x_max],
            'y_range': [y_min, y_max],
            'z_range': [z_min, z_max]
        }

        if uuid:
            job_listing = distance_df.query('wv_uuid == @uuid')
            if not job_listing.empty:
                text = (x_text, y_text, z_text)
                distances = (job_listing.iloc[0]['x_dist'], job_listing.iloc[0]['y_dist'], job_listing.iloc[0]['z_dist'])
                alignment_summary = gen_utils.generate_alignment_summary(uuid, text, distances,dist_ranges, client = weaviate_client)
                return JsonResponse({'summary': alignment_summary})

    return JsonResponse({'summary': None})