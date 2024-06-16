import weaviate
from pprint import pprint
def generate_job_summary_prompt(text, distances, dist_ranges):
    x_text, y_text, z_text = text
    x_dist, y_dist, z_dist = distances

    prompt = f"""
    Analyze the job listing attached to this query.

    The job listing is measured on three axes: {x_text}, {y_text}, and {z_text}. The cosine distances of the job listing on these axes are as follows:

    - {x_text}: {x_dist:.3f}
    - {y_text}: {y_dist:.3f}
    - {z_text}: {z_dist:.3f}

    A smaller cosine distance indicates better alignment, while a larger cosine distance indicates worse alignment.

    The ranges for the cosine distances are as follows:

    - {x_text}: {dist_ranges['x_range'][0]:.3f} to {dist_ranges['x_range'][1]:.3f}
    - {y_text}: {dist_ranges['y_range'][0]:.3f} to {dist_ranges['y_range'][1]:.3f}
    - {z_text}: {dist_ranges['z_range'][0]:.3f} to {dist_ranges['z_range'][1]:.3f}

    Use these ranges along with the provided distances to evaluate the job listing's alignment with each axis.

    Based on the provided job listing and its cosine distances on the three axes, please generate a summary that describes how well the job listing aligns with each axis. Provide insights into the relevance and significance of the job listing in relation to the axes.

    Consider the following questions in your summary:

    1. How closely does the job listing match the characteristics and requirements of each axis?
    2. What are the key aspects of the job listing that contribute to its alignment or misalignment with each axis?
    3. Are there any notable strengths or weaknesses of the job listing in relation to the axes?
    4. How do the cosine distances reflect the overall fit of the job listing to the axes?

    Remember the following:

    1. Higher cosine distance means that the job listing is less aligned with the axis.
    2. Lower cosine distance means that the job listing is more aligned with the axis.
    3. Consider the key features and requirements of the job listing when evaluating its alignment with each axis.
    4. Never output a number with more than 3 decimal places.
    5. When referring to an axis, please use the following format: <strong>{x_text}</strong>, <strong>{y_text}</strong>, or <strong>{z_text}</strong>.

    Response:

    Your response should have 3 sections:
    1. An overall evaluation of the job listing.
    2. A summary of the alignment of the job listing with each axis.
    3. A summary of the overall alignment of the job listing with the axes.

    Please format your response in HTML using the following structure:

    <div class="job-summary">
      <h3>Overall Evaluation</h3>
      <p>[Provide an overall evaluation of the job listing based on its alignment with the axes]</p>

      <h3>Axis Alignment</h3>
      <ul>
        <li><strong>{x_text}</strong>: [Describe the alignment of the job listing with the {x_text} axis]</li>
        <li><strong>{y_text}</strong>: [Describe the alignment of the job listing with the {y_text} axis]</li>
        <li><strong>{z_text}</strong>: [Describe the alignment of the job listing with the {z_text} axis]</li>
      </ul>

      <h3>Overall Alignment</h3>
      <p>[Provide a summary of the overall alignment of the job listing with the axes, considering the cosine distances and the key aspects of the job listing]</p>
    </div>
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
