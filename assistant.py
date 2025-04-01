import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
ASSISTANT_ID = os.getenv("DAISY_ASSISTANT_ID")

def create_or_continue_thread(thread_id, user_message):
    if not thread_id:
        thread = openai.beta.threads.create()
        thread_id = thread.id

    openai.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=user_message
    )

    run = openai.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=ASSISTANT_ID,
    )

    while run.status in ['queued', 'in_progress']:
        run = openai.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)

    messages = openai.beta.threads.messages.list(thread_id)
    latest_message = messages.data[0].content[0].text.value

    return thread_id, latest_message
