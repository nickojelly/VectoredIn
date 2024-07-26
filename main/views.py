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
from pprint import pprint
import numpy as np
from .models import JobPosting, QueryEmbedding, JobListing, SubComponemtEmbeddings  # Import the JobPosting model
import re
import textwrap
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
        vector,_ = gen_utils.vectorize(openai_client,weaviate_client, text)
        vector_rag, rag_query = gen_utils.vectorize(openai_client,weaviate_client, text, rag=True)
        query_embedding = QueryEmbedding(query=text, query_embedding=json.dumps(vector),rag_query=rag_query, rag_query_embedding=json.dumps(vector_rag))
        query_embedding.save()
    else:
        # If the query embedding exists, retrieve the vectors
        # print(f"Query embedding exists for query: {text}, query = {query_embedding.rag_query}")
        if rag:
            vector = np.array(json.loads(query_embedding.rag_query_embedding))
        else:
            vector = np.array(json.loads(query_embedding.query_embedding))

    
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
    listings = JobListing.objects.filter(wv_uuid__in=distance_df['uuid'])
    df = pd.DataFrame(list(listings.values()))
    distance_df = distance_df.merge(df, left_on='uuid', right_on='wv_uuid')


    return distance_df

def generate_plot(x_text, y_text, z_text,k=10,n=5, update=False,rag=False):

    #Process the data:
    if update:
        distance_df = update_data(x_text, y_text, z_text,k,n,rag)
    else: 
        distance_df_path = os.path.join(settings.BASE_DIR, 'static', 'myapp', 'distance_df.fth')
        distance_df = pd.read_feather(distance_df_path)
    # distance_df = distance_df.sample(n=10)
    red_point = distance_df.query('uuid == "00bf8c9d-be8f-45f3-9748-4a44379745e4"')
    
    # distance_df['sum_dist'] = distance_df['x_dist'] + distance_df['y_dist'] + distance_df['z_dist']
    # distance_df = distance_df.sort_values(by='sum_dist', ascending=False)
    x = distance_df['x_dist'].values
    y = distance_df['y_dist'].values
    z = distance_df['z_dist'].values
    custom_data = distance_df.wv_uuid.values
    # print(custom_data)

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
            opacity=1,
                    colorbar=dict(
            title='Semantic Distance',
            x=0.85,
            y=0.5,
            len=0.75,
        )
        ),
        text=distance_df['title']+' | '+distance_df['company_name'],
        hoverinfo='text',
        name='Semantic Distance',
        customdata=custom_data,
        showlegend=False
    )

    # camera = dict(
    #     eye=dict(
    #         x=2,
    #         y=2,
    #         z=2
    #     )
    # )

    layout = dict(
        autosize=True,
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
        # width=800,  # Adjust the width as needed
        # height=600,  # Adjust the height as needed
        margin=dict(r=10, l=10, b=0, t=10),
        # scene_camera=camera,
    )

    plot_div = plot({'data': [trace], 'layout': layout}, output_type='div', config={'responsive': True})

    fig = go.Figure(data=[trace], layout=layout)
    # plot(fig, auto_open=False, filename='fullplot.html')

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

import re

def parse_asterisks(text):
    parts = text.split('**')
    for i in range(1, len(parts), 2):
        parts[i] = f'<strong>{parts[i]}</strong>'
    return ''.join(parts)

def apply_formatting(text):
    text = parse_asterisks(text)

    # Split text into sections
    sections = re.split(r'\n\s*\n', text)
    
    formatted_sections = []
    for section in sections:
        lines = section.split('\n')
        if all(line.strip().startswith('•') for line in lines if line.strip()):
            # This is a list
            list_items = [f'<li>{line.strip()[1:].strip()}</li>' for line in lines if line.strip()]
            formatted_sections.append(f'<ul>{"".join(list_items)}</ul>')
        else:
            # This is a regular paragraph or a header
            formatted_lines = []
            for line in lines:
                line = line.strip()
                if line:
                    if line.isupper():
                        formatted_lines.append(f'<h3>{line}</h3>')
                    else:
                        formatted_lines.append(f'<p>{line}</p>')
            formatted_sections.append(''.join(formatted_lines))
    
    return ''.join(formatted_sections)

def format_description_with_ner(text, entities):
    formatted_text = text
    offset = 0
    # print(f"Entities = {entities}")
    # print(f"text = {text}")
    for entity in sorted(entities, key=lambda x: x['start'], reverse=True):
        start = entity['start'] + offset
        end = entity['end'] + offset
        span = f'<span class="ner-{entity["label"].lower()}">{formatted_text[start:end]}</span>'
        formatted_text = formatted_text[:start] + span + formatted_text[end:]
        # offset += len(span) - (end - start)
    
    return formatted_text

@csrf_exempt
def get_related_roles(request):
    if request.method == 'POST':
        point_data = json.loads(request.body)
        uuid = point_data['customData']
        
        job_listing = JobListing.objects.filter(wv_uuid=uuid).first()
        vector = json.loads(job_listing.vector)
        search_results = hnsw_obj.search(q=vector, k=6)
        
        remapped = [[hnsw_obj.df['wv_uuid'].iloc[i[0]], hnsw_obj.data[i[0]], i[1]] for i in search_results[1:]]
        related_df = pd.DataFrame(data=remapped, columns=['wv_uuid', 'vector', 'dist'])
        
        related_roles = JobListing.objects.filter(wv_uuid__in=related_df['wv_uuid'].values)
        df = pd.DataFrame(list(related_roles.values()))
        related_df = pd.merge(related_df, df, on='wv_uuid')
        
        # Generate HTML for related roles
        html_content = generate_related_roles_html(related_df)
        
        return JsonResponse({'html': html_content})

def generate_related_roles_html(related_df):
    html = ""
    for _, role in related_df.iterrows():
        html += f"""
        <div class="box">
            <div class="columns">
                <div class="column">
                    <h5 class="title is-5">{role['title']}</h5>
                    <p class="subtitle is-6">{role['company_name']}</p>
                </div>
                <div class="column is-narrow">
                    <div class="box has-background-light">
                        <p class="has-text-weight-bold">Distance</p>
                        <p>{role['dist']:.5f}</p>
                    </div>
                </div>
            </div>
        </div>
        """
    return html


def truncate_text(text, max_length=100):
    return text[:max_length] + '...' if len(text) > max_length else text

@csrf_exempt
def get_point_calculations(request):
    if request.method == 'POST':
        point_data = json.loads(request.body)
        pprint(point_data)
        uuid = point_data['customData']
        x = point_data['x']
        y = point_data['y']
        z = point_data['z']


        job_listing = JobListing.objects.filter(wv_uuid = uuid).first()
        vector = json.loads(job_listing.vector)
        
        entity_indicies = json.loads(job_listing.entity_indices)

        original_vector_df = pd.DataFrame({
            'entity': ['Average Embedding'],
            'text': ['Vector'],
            'embedding': [vector],
            'index': [-1]  # Use a unique index for the original vector
        })

        entities = SubComponemtEmbeddings.objects.filter(index__in=entity_indicies)

        entities = pd.DataFrame(list(entities.values()))

        entities.embedding = entities['embedding'].apply(lambda x: np.frombuffer(x, dtype=np.float32))

        entities = pd.concat([entities, original_vector_df], ignore_index=True)

        x_vector  = vectorize_query_db(point_data['xText'],True)
        y_vector  = vectorize_query_db(point_data['yText'],True)
        z_vector  = vectorize_query_db(point_data['zText'],True)

        entities['x_dist'] = entities['embedding'].apply(lambda x: hnsw_obj.distance(x,x_vector))
        entities['y_dist'] = entities['embedding'].apply(lambda x: hnsw_obj.distance(x,y_vector))
        entities['z_dist'] = entities['embedding'].apply(lambda x: hnsw_obj.distance(x,z_vector))

        x = entities['x_dist'].values
        y = entities['y_dist'].values
        z = entities['z_dist'].values
        # custom_data = distance_df.wv_uuid.values
        # print(custom_data)
        
        original_vector = entities.iloc[-1]
        entities = entities.iloc[:-1]
        global_min = min(x.min(), y.min(), z.min())
        global_max = max(x.max(), y.max(), z.max())

        print(f"uuid = {uuid}")
        # print(f"vector = {vector}")
        min_x = min(point_data['ranges']['x_range'][0], x.min())
        min_y = min(point_data['ranges']['y_range'][0], y.min())
        min_z = min(point_data['ranges']['z_range'][0], z.min())
        global_min = min(min_x,min_y,min_z)
        global_max = max(x.max(), y.max(), z.max())

        trace = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
            size=8,
            color=x+y+z,
            colorscale='Viridis',
            opacity=1,
                    colorbar=dict(
            title='Semantic Distance',
            x=0.85,
            y=0.5,
            len=0.75,
        ),),
            showlegend=False,
            text=[truncate_text(f"{entity} : {text}") for entity, text in zip(entities['entity'], entities['text'])],
            hoverinfo='text',
            name='Sub Component Distances',
            # customdata=custom_data
        )

        original_vector_trace = go.Scatter3d(
            x=[original_vector['x_dist']],
            y=[original_vector['y_dist']],
            z=[original_vector['z_dist']],
            mode='markers',
            marker=dict(
                size=8,
                color='red',
                symbol='diamond',
            ),
            text=[f"{original_vector['entity']}"],
            hoverinfo='text',
            name='Average Embedding Distance',
            showlegend=False
        )

        layout = dict(
            # reponsive=True,
            autosize=True,
            scene=dict(
                aspectmode='cube',
                xaxis=dict(
                    range=[global_min, global_max],
                    title=point_data['xText'],
                    backgroundcolor="rgb(200, 230, 225)",  # Muted color for x-axis
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white",
                ),
                yaxis=dict(
                    range=[global_min, global_max],
                    title=point_data['yText'],
                    backgroundcolor="rgb(220, 230, 240)",  # Muted color for y-axis
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white",
                ),
                zaxis=dict(
                    range=[global_min, global_max],
                    title=point_data['zText'],
                    backgroundcolor="rgb(230, 240, 220)",  # Muted color for z-axis
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white",
                ),
            ),
            hovermode='closest',
            # width=800,  # Adjust the width as needed
            # height=600,  # Adjust the height as needed
            margin=dict(r=10, l=10, b=10, t=10),
            scene_camera = dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=2.5, y=2.5, z=2.5)
            )
        )

        plot_div = plot({'data': [trace, original_vector_trace], 'layout': layout}, output_type='div', config={'responsive': True})
        # print('returing')
        fig = go.Figure(data=[trace, original_vector_trace], layout=layout)
        plot_data_json = json.dumps([trace, original_vector_trace], cls=PlotlyJSONEncoder)
        response_data = {
            'data': plot_data_json, 
            'layout': layout,
        }

        return JsonResponse(response_data)

        return plot_div, [trace], layout, 
from plotly.offline import plot
@csrf_exempt
def get_point_highlight(request):
    if request.method == 'POST':
        point_data = json.loads(request.body)
        print(point_data)
        uuid = point_data['customData']
        x = point_data['x']
        y = point_data['y']
        z = point_data['z']
        highlight_trace = go.Scatter3d(
            x=[x],
            y=[y],
            z=[z],
            mode='markers',
            marker=dict(
                size=8,
                color='red',
                symbol='diamond',
            ),
            text=[point_data['text']],
            hoverinfo='text',
            name='Average Embedding Distance',
            showlegend=False,
            customdata=[point_data['customData']]
        )

        highlight_trace = json.dumps([highlight_trace], cls=PlotlyJSONEncoder)

        return JsonResponse({'highlight_trace': highlight_trace})


@csrf_exempt
def get_point_summary(request):
    if request.method == 'POST':
        point_data = json.loads(request.body)
        uuid = point_data['customData']
        x = point_data['x']
        y = point_data['y']
        z = point_data['z']

        job_listing = JobListing.objects.filter(wv_uuid = uuid).first()

        # job_listing = listing_df.query('wv_uuid == @uuid')
        text = job_listing.text
        annotations = json.loads(job_listing.annotations)
        title = job_listing.title
        company_name = job_listing.company_name


        # print(f"Annotations = {annotations}, type {type(annotations)}, text = {text}")
        # highlight_trace = go.Scatter3d(
        #     x=[x],
        #     y=[y],
        #     z=[z],
        #     mode='markers',
        #     marker=dict(
        #         size=8,
        #         color='red',
        #         symbol='diamond',
        #     ),
        #     text=[f"{title} | {company_name}"],
        #     hoverinfo='text',
        #     name='Average Embedding Distance',
        #     showlegend=False
        # )

        # highlight_trace = json.dumps([highlight_trace], cls=PlotlyJSONEncoder)


        formatted_text = format_description_with_ner(text, annotations)

        final_formatted_text = apply_formatting(formatted_text)

        summary_data = {
            'title': job_listing.title,
            'company': job_listing.company_name,
            'location': "NA",
            'description': text,
            'formatted_description': final_formatted_text ,
            'x': x,
            'y': y,
            'z': z
        }
        
        return JsonResponse({'summary': summary_data})
        # else:
            # return JsonResponse({'summary': None})
        


@csrf_exempt
def generate_plot_summary(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        # uuid = data.get('uuid')
        x_text = data.get('x_text', '')
        y_text = data.get('y_text', '')
        z_text = data.get('z_text', '')

        plotdata = data.get('plot_data', [])
        uuids = plotdata[0]['customdata']
        x_data = plotdata[0]['x']
        y_data = plotdata[0]['y']
        z_data = plotdata[0]['z']

        # print(f"Plotdata = {plotdata}")

        distance_df_path = os.path.join(settings.BASE_DIR, 'static', 'myapp', 'distance_df.fth')
        distance_df = pd.read_feather(distance_df_path)

        min_x_uuid = uuids[np.argmin(x_data)]
        min_y_uuid = uuids[np.argmin(y_data)]
        min_z_uuid = uuids[np.argmin(z_data)]

        uuids = [min_x_uuid, min_y_uuid, min_z_uuid]

        text = (x_text, y_text, z_text)




        plot_summary = gen_utils.generate_plot_summary(uuids,text, client = weaviate_client, openai_client=openai_client)
        return JsonResponse({'summary': plot_summary})

    return JsonResponse({'summary': None})
    
@csrf_exempt
def get_alignment_summary(request):
    if request.method == 'POST':
        data = json.loads(request.body)

        # print('allignment summary called')
        # pprint(data)
        uuid = data.get('uuid')
        x_text = data.get('x_text', '')
        y_text = data.get('y_text', '')
        z_text = data.get('z_text', '')

        dist_ranges = data['pointdata']['ranges']
        

        if uuid:
            text = (x_text, y_text, z_text)
            distances = (data['pointdata']['x'], data['pointdata']['y'], data['pointdata']['z'])
            alignment_summary = gen_utils.generate_alignment_summary(uuid, text, distances,dist_ranges, client = weaviate_client)
            return JsonResponse({'summary': alignment_summary})
        else:
            print('no uuid')
    return JsonResponse({'summary': None})