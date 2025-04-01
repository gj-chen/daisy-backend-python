import os
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from assistant import create_or_continue_thread
from image_filter import filter_images
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)

CORS(app)  # Enables CORS clearly for all routes

GOOGLE_AGENT_URL = os.getenv("GOOGLE_AGENT_URL")

@app.route('/')
def home():
    return "Daisy backend is running!"

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('userMessage')
    thread_id = data.get('threadId', None)

    print("[LOG] Received request:", data)

    thread_id, assistant_reply = create_or_continue_thread(thread_id, user_message)
    print("[LOG] Daisy Assistant reply:", assistant_reply)

    onboarding_complete = "Onboarding complete." in assistant_reply
    print("[LOG] Onboarding complete status:", onboarding_complete)

    if onboarding_complete:
        if "Onboarding complete. Here's your search query:" in assistant_reply:
            search_query = assistant_reply.split("Onboarding complete. Here's your search query:")[1].strip()
            print("[LOG] Extracted search query:", search_query)
        else:
            print("[ERROR] Onboarding completion phrase found, but query extraction failed.")
            search_query = None

        if search_query:
            images_response = requests.get(
                f"{GOOGLE_AGENT_URL}/search-google-images",
                params={"q": search_query}
            ).json()
            print("[LOG] Images fetched from Google Agent:", images_response)

            images = images_response.get('images', [])
            filtered_images = filter_images(images, search_query)
            print("[LOG] Images after GPT-4 Vision filtering:", filtered_images)

            response = {
                "threadId": thread_id,
                "onboarding": False,
                "moodboard": {
                    "imageUrls": filtered_images,
                    "rationale": {
                        "goal": f"Curated visuals based on '{search_query}'.",
                        "whatWorks": "These images align with your described vibe.",
                        "avoid": "Avoid styles conflicting with your aesthetic.",
                        "tip": "Use these visuals as inspiration for cohesive styling."
                    }
                }
            }
            print("[LOG] Final response:", response)
            return jsonify(response)
        else:
            print("[ERROR] Search query was None. Not proceeding to image search.")

    response = {
        "threadId": thread_id,
        "onboarding": True,
        "message": assistant_reply
    }
    print("[LOG] Onboarding response:", response)
    return jsonify(response)

if __name__ == '__main__':
    print("[LOG] Starting Daisy backend...")
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 8000)))
