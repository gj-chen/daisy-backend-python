import openai
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def load_prompt():
    prompt_path = os.path.join(os.path.dirname(__file__), 'prompts', 'daisy_assistant_prompt.txt')
    with open(prompt_path, 'r') as file:
        return file.read()

def create_daisy_assistant():
    instructions = load_prompt()

    assistant = openai.beta.assistants.create(
        name="Daisy – Your AI Stylist & Creative Director",
        instructions=instructions,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "search_curated_images",
                    "description": "Search curated fashion images from Daisy’s internal moodboard database",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "A short 1–2 sentence style summary to guide the image search"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ],
        model="gpt-4-1106-preview"
    )

    print("✅ Assistant created:")
    print("ID:", assistant.id)


if __name__ == "__main__":
    create_daisy_assistant()
