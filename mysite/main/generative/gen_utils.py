import weaviate
from pprint import pprint
def generate_job_summary_prompt(text, distances, dist_ranges):
    x_text, y_text, z_text = text
    x_dist, y_dist, z_dist = distances


    prompt = f"""
    Analyze the job listing attached to this query.

    The job listing is measured on three axes: {x_text}, {y_text}, and {z_text}. The cosine distances of the job listing on these axes are as follows:

    - {x_text}: {x_dist}
    - {y_text}: {y_dist}
    - {z_text}: {z_dist}

    A smaller cosine distance is better alignment. A larger cosine distance is worse alignment.

    The ranges for the cosine distances are as follows:

    - {x_text}: {dist_ranges['x_range'][0]} to {dist_ranges['x_range'][1]}
    - {y_text}: {dist_ranges['y_range'][0]} to {dist_ranges['y_range'][1]}
    - {z_text}: {dist_ranges['z_range'][0]} to {dist_ranges['z_range'][1]}

    Use these ranges along with the provided distance to evaluate the job listing's alignment with each axis.


    Based on the provided job listing and its cosine distances on the three axes, please generate a summary that describes how well the job listing aligns with each axis. Provide insights into the relevance and significance of the job listing in relation to the axes.

    Consider the following questions in your summary:

    1. How closely does the job listing match the characteristics and requirements of each axis?
    2. What are the key aspects of the job listing that contribute to its alignment or misalignment with each axis?
    3. Are there any notable strengths or weaknesses of the job listing in relation to the axes?
    4. How do the cosine distances reflect the overall fit of the job listing to the axes?

    Remember to following:

    1. Higher Cosine distance means that the job listing is less aligned with the axis.
    2. Lower Cosine distance means that the job listing is more aligned with the axis.
    3. Consider the key features and requirements of the job listing when evaluating its alignment with each axis.
    4. Never put out a number with more than 3 decimal places.
    5. If you are refering to and axis, please refer to the axis as {x_text}, {y_text}, or {z_text}, and return it in bold.

    Response:

    Your response should have 3 section, the first is an overall evalutation of the job listing, the second is a summary of the alignment of the job listing with each axis, and the third is a summary of the overall alignment of the job listing with the axes.

    Please format your response as html:

    """

    return prompt

def generate_alignment_summary(uuid,  text, distances,dist_ranges, client: weaviate.Client):
    print("Starting alignment summary")

    prompt = generate_job_summary_prompt( text, distances, dist_ranges)

    pprint(f"prompt generated : {prompt}")

    listings = client.collections.get("JobListings")

    job_weaviate = listings.query.fetch_object_by_id(uuid, include_vector=True)

    single_query = listings.generate.near_vector(near_vector=job_weaviate.vector['default'], limit=1, grouped_task=prompt)

    return single_query.generated
