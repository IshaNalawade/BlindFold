import torch
from transformers import pipeline

def summary_resume(content):
    job_description = read_text_from_file(job_description_file)
    answer_1 = read_text_from_file(answer_1_file)

    # Create prompt for relevance evaluation
    prompt = f"""
Job Description: {job_description}

Interview Question: Tell us about a time when you had to handle a difficult client.

Candidate's Answer: {answer_1}

Instructions:
1. Evaluate how well the candidate's answer aligns with the job description.
2. Assess the relevance of the candidate's answer to the interview question.
3. Consider the following criteria:
    - How effectively did the candidate handle the difficult client?
    - Does the answer demonstrate skills relevant to a sales role, such as problem-solving, empathy, effective communication, and the ability to maintain a positive relationship?
    - How well does the candidate’s answer reflect the specific needs of the sales position described in the job description?
4. Provide a relevance score from 1 to 10, where 10 means the answer is highly relevant and well-suited for the role.
5. Provide a brief explanation of why you gave this score, touching on specific strengths or areas for improvement in the answer.
"""

    # Create the model
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )

    # Start a chat session and send the prompt
    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(prompt)

    relevance_response = response.text
    html_response = markdown.markdown(relevance_response)
    return html_response
    return
pipe = pipeline(
    "text-generation",
    model="google/gemma-2-2b-it",
    device="mps",  # replace with "cuda" to run on a windows device
)

text = "Once upon a time,"
outputs = pipe(text, max_new_tokens=500)
response = outputs[0]["generated_text"]
print(response)
# from huggingface_hub import login

# login(token="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")