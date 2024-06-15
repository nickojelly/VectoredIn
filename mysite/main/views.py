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
from django.template.loader import render_to_string
from django.template import RequestContext
import traceback
import logging
from plotly.utils import PlotlyJSONEncoder
import json
from .utils import hnsw_obj,openai_client,listing_df, vectorize
from django.views.decorators.csrf import csrf_exempt
logger = logging.getLogger(__name__)


def homepage(request):
    return HttpResponse('Hello, World!')

def update_plot_data(request, x_text, y_text, z_text):
    return get_plot_data(request, x_text, y_text, z_text)

def get_plot_data(request, x_text, y_text, z_text):
    plot_div, plot_data, plot_layout = generate_plot(x_text, y_text, z_text)

    # Convert plot_data to a JSON-serializable format
    plot_data_json = json.dumps(plot_data, cls=PlotlyJSONEncoder)

    test_hnsw(request)

    response_data = {
        'data': plot_data_json,
        'layout': plot_layout,
    }

    print(plot_layout)

    return JsonResponse(response_data)


def update_data(x_text, y_text, z_text):
    #Process the data:
    pass
    # x_vector = vectorize(openai_client,x_text)
    # y_vector = vectorize(openai_client,y_text)
    # z_vector = vectorize(openai_client,z_text)

    # x_list = hnsw_obj.serach_along_axis(x_vector, 10)
    # y_list = hnsw_obj.serach_along_axis(y_vector, 10)
    # z_list = hnsw_obj.serach_along_axis(z_vector, 10)
    # full_list = x_list|y_list|z_list

    # distance_list = []
    # for uuid,array in full_list.items():
    #     print(uuid, array)
    #     x_dist = hnsw_obj.distance(x_vector, array)
    #     y_dist = hnsw_obj.distance(y_vector, array)
    #     z_dist = hnsw_obj.distance(z_vector, array)
    #     distance_list.append((uuid, x_dist, y_dist, z_dist))

    # distance_df = pd.DataFrame(data=distance_list, columns=['uuid', 'x_dist', 'y_dist', 'z_dist'])
    # distance_df = distance_df.merge(listing_df, left_on='uuid', right_on='wv_uuid')

    # distance_df.to_feather(os.path.join(settings.BASE_DIR, 'static', 'myapp', 'distance_df.fth'))

def generate_plot(x_text, y_text, z_text):
    #Process the data:
    update_data(x_text, y_text, z_text)
    
    distance_df_path = os.path.join(settings.BASE_DIR, 'static', 'myapp', 'distance_df.fth')
    distance_df = pd.read_feather(distance_df_path)
    # distance_df = distance_df.sample(n=10)
    x = distance_df['x_dist'].values
    y = distance_df['y_dist'].values
    z = distance_df['z_dist'].values
    custom_data = distance_df.wv_uuid.values
    global_min = min(x.min(), y.min(), z.min())
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
            opacity=0.8,
            colorbar=dict(title='Semantic Distance')
        ),
        text=distance_df['title']+' | '+distance_df['company_name'],
        hoverinfo='text',
        name='Semantic Distance',
        customdata=custom_data
    )

    layout = dict(
        # title='t-SNE plot',
        scene=dict(
            aspectmode='cube',
            xaxis=dict(range=[global_min, global_max],title=x_text),
            yaxis=dict(range=[global_min, global_max],title=y_text),
            zaxis=dict(range=[global_min, global_max],title=z_text),
        ),
        hovermode='closest',
        width=800,  # Adjust the width as needed
        height=800,  # Adjust the height as needed
        
    )

    plot_div = plot({'data': [trace], 'layout': layout}, output_type='div', config={'responsive': True})

    return plot_div, [trace], layout


def test_hnsw(request):
    result = hnsw_obj.serach_along_axis(q=hnsw_obj.data[0],k=5)
    print(f"Result of search = {result}")



def plot_view(request):
    initial_x_text = 'Machine Learning Engineer'
    initial_y_text = 'Data Scientist'
    initial_z_text = 'Accountant'

    form = PlotForm()

    plot_div, plot_data, plot_layout = generate_plot(initial_x_text, initial_y_text, initial_z_text)

    context = {
        'form': form,
        'initial_x_text': initial_x_text,
        'initial_y_text': initial_y_text,
        'initial_z_text': initial_z_text,
        'plot_div': plot_div,
        'plot_data': json.dumps(plot_data, cls=PlotlyJSONEncoder),
        'plot_layout': plot_layout,
    }

    return render(request, "your_template.html", context)

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
        # Generate the plot summary based on the current plot data
        plot_summary = "This is a general summary of the plot. You can include any relevant information here."
        
        return JsonResponse({'summary': plot_summary})