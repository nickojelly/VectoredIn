import weaviate
from pprint import pprint

import openai
from ..models import JobPosting, QueryEmbedding  # Import the JobPosting model

def generate_axis_summary_prompt(text):
    prompt = f"""These job listings are the top 3 most aligned with the "{text}" axis based on their cosine distances.

Please provide a concise summary that highlights the key aspects of these job listings 
that contribute to their alignment with the "{text}" axis. Consider the following points in your summary:

1. What are the common themes, skills, or requirements mentioned across these job listings?
2. How do these job listings relate to the concept of "{text}"?
3. Are there any specific qualifications, experience, or responsibilities that make these job listings highly relevant to "{text}"?

Please format your response as follows:

Summary: [A concise summary of the key aspects that make these job listings aligned with the "{text}" axis]

Common Themes:
- [Theme 1]
- [Theme 2]
- [Theme 3]

Relevance to "{text}":
- [Point 1]
- [Point 2]
- [Point 3]

Specific Qualifications or Experience:
- [Qualification 1]
- [Qualification 2]
- [Qualification 3]"""
    
    return prompt

def generate_plot_summary_prompt(axis, text):
    prompt = f"""
    Based on the summaries of the top 3 job listings for each axis, provide an overview of the similarities and differences between the axes.

    Axis Summaries:

    <article class="message is-primary">
        <div class="message-header">
            <p><strong>{axis[0]}</strong></p>
        </div>
        <div class="message-body">
            <p>{text[0]}</p>
        </div>
    </article>

    <article class="message is-info">
        <div class="message-header">
            <p><strong>{axis[1]}</strong></p>
        </div>
        <div class="message-body">
            <p>{text[1]}</p>
        </div>
    </article>

    <article class="message is-success">
        <div class="message-header">
            <p><strong>{axis[2]}</strong></p>
        </div>
        <div class="message-body">
            <p>{text[2]}</p>
        </div>
    </article>

    Please analyze the summaries and highlight the key similarities and differences across the axes. Consider the following points in your overview:

    1. What common themes, skills, or requirements are shared among the axes?
    2. Are there any notable differences in the focus or emphasis of each axis?
    3. How do the specific qualifications or experience mentioned in each axis compare?
    4. Are there any unique aspects or characteristics that distinguish one axis from the others?

    Please format your response in HTML using the following structure:

    <div class="axis-summary">
        <p>[Provide a concise overview of the similarities and differences between the axes based on the summaries]</p>

        <article class="message is-warning">
            <div class="message-header">
                <p><strong>Similarities</strong></p>
            </div>
            <div class="message-body">
                <ul>
                    <li>[Similarity 1]</li>
                    <li>[Similarity 2]</li>
                    <li>[Similarity 3]</li>
                </ul>
            </div>
        </article>

        <article class="message is-danger">
            <div class="message-header">
                <p><strong>Differences</strong></p>
            </div>
            <div class="message-body">
                <ul>
                    <li>[Difference 1]</li>
                    <li>[Difference 2]</li>
                    <li>[Difference 3]</li>
                </ul>
            </div>
        </article>

        <article class="message is-light">
            <div class="message-header">
                <p><strong>Unique Aspects</strong></p>
            </div>
            <div class="message-body">
                <ul>
                    <li><strong>{axis[0]}</strong>: [Unique aspect]</li>
                    <li><strong>{axis[1]}</strong>: [Unique aspect]</li>
                    <li><strong>{axis[2]}</strong>: [Unique aspect]</li>
                </ul>
            </div>
        </article>

        <p><strong>Comparison of Qualifications or Experience:</strong></p>
        <p>[A brief comparison of the specific qualifications or experience mentioned in each axis]</p>
    </div>
    """
    return prompt

def generate_job_summary_prompt(text, distances, dist_ranges):
    x_text, y_text, z_text = text
    x_dist, y_dist, z_dist = distances

    print(dist_ranges)

    x_percentile = 100 * (1-(x_dist - dist_ranges['x_range'][0]) / (dist_ranges['x_range'][1] - dist_ranges['x_range'][0]))
    y_percentile = 100 * (1-(y_dist - dist_ranges['y_range'][0]) / (dist_ranges['y_range'][1] - dist_ranges['y_range'][0]))
    z_percentile = 100 * (1-(z_dist - dist_ranges['z_range'][0]) / (dist_ranges['z_range'][1] - dist_ranges['z_range'][0]))

    print(f"{x_percentile=}, {y_percentile=}, {z_percentile=}")

    prompt = f"""
    Analyze the job listing asigned to this prompt.

    The job listing is measured on three axes: {x_text}, {y_text}, and {z_text}. The cosine distances of the job listing on these axes are as follows:

    - {x_text}: {x_dist:.3f} which is the {x_percentile:.1f}% percentile of all job listings.
    - {y_text}: {y_dist:.3f} which is the {y_percentile:.1f}% percentile of all job listings.
    - {z_text}: {z_dist:.3f} which is the {z_percentile:.1f}% percentile of all job listings.

    These are the ranges you should use for your response:

    - [0-20%] Misaligned
    - [20-40%] Slightly Misaligned
    - [40-60%] Aligned
    - [60-80%] Slightly Aligned
    - [80-100%] Very Aligned

    Always reference these when talking about alignment, very important when talking about the alignment of job listings.

    A smaller cosine distance indicates better alignment, while a larger cosine distance indicates worse alignment.

    Based on the provided job listing and its cosine distances on the three axes, please generate a summary that describes how well the job listing aligns with each axis. Provide insights into the relevance and significance of the job listing in relation to the axes.

    Consider the following questions in your summary:

    1. What are the expected characteristics and requirements for each axis ({x_text}, {y_text}, {z_text})? For example:
       - {x_text}: Machine Learning Visualizations, Data Forecasting, etc.
       - {y_text}: Statistical Analysis, Data Mining, etc.
       - {z_text}: Customer Service, Hospitality, etc.
    2. What are the main points of the job listing, and what are these points similar to?
    3. How closely does the job listing match the characteristics and requirements of each axis?
    4. What are the key aspects of the job listing that contribute to its alignment or misalignment with each axis?
    5. Are there any notable strengths or weaknesses of the job listing in relation to the axes?
    6. How do the cosine distances reflect the overall fit of the job listing to the axes?

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
      <p>[Provide an overall evaluation of the job listing based on its alignment with the axes. Include insights about the job, key skills, and common expectations for similar roles.]</p>

      <article class="message is-primary">
        <div class="message-header">
          <p><strong>{x_text}</strong></p>
        </div>
        <div class="message-body">
          <p><em>[Describe the alignment of the job listing with the {x_text} axis. Include reasons for alignment or misalignment, even if it doesn't align well.]</em></p>
        </div>
      </article>
      <article class="message is-info">
        <div class="message-header">
          <p><strong>{y_text}</strong></p>
        </div>
        <div class="message-body">
          <p><em>[Describe the alignment of the job listing with the {y_text} axis. Include reasons for alignment or misalignment, even if it doesn't align well.]</em></p>
        </div>
      </article>
      <article class="message is-success">
        <div class="message-header">
          <p><strong>{z_text}</strong></p>
        </div>
        <div class="message-body">
          <p><em>[Describe the alignment of the job listing with the {z_text} axis. Include reasons for alignment or misalignment, even if it doesn't align well.]</em></p>
        </div>
      </article>

      <p>[Provide a summary of the overall alignment of the job listing with the axes, considering the key aspects of the job listing.]</p>
      <p><strong>Key Skills and Responsibilities:</strong></p>
      <ul>
        <li>[List key skills and responsibilities mentioned in the job listing]</li>
      </ul>
    </div>
    """

    return prompt

def generate_plot_summary(uuid,  text,  client: weaviate.Client, openai_client: openai.OpenAI):
    print("Starting alignment summary")
    uuid_x, uuid_y, uuid_z = uuid
    text_x, text_y, text_z = text

    prompts = []
    summaries = []
    for i in range(3):
        prompt = generate_axis_summary_prompt( text[i])

        pprint(f"prompt generated : {prompt}")
        prompts.append(prompt)

        listings = client.collections.get("JobListings")
        job_weaviate = listings.query.fetch_object_by_id(uuid[i], include_vector=True)
        single_query = listings.generate.near_vector(near_vector=job_weaviate.vector['default'], limit=3, grouped_task=prompt)
        pprint(f"summary query generated : {single_query}")
        summaries.append(single_query)

    plot_summary_prompt = generate_plot_summary_prompt(text, prompts)

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": plot_summary_prompt}
        ],
        max_tokens=500,
        temperature=0.7,
    )

    msg = response.choices[0].message.content
    print(msg)

    return msg



def generate_alignment_summary(uuid,  text, distances,dist_ranges, client: weaviate.Client):
    print("Starting alignment summary")

    prompt = generate_job_summary_prompt( text, distances, dist_ranges)

    pprint(f"prompt generated : {prompt}")

    listings = client.collections.get("JobListings")

    job_weaviate = listings.query.fetch_object_by_id(uuid, include_vector=True)

    single_query = listings.generate.near_vector(near_vector=job_weaviate.vector['default'], limit=1, grouped_task=prompt)

    pprint(f"single query generated : {single_query}")

    return single_query.generated
