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

    if 'conversationQueue' not in data:
        data['conversationQueue'] = []

    conversation_queue = data['conversationQueue']

    if conversation_queue:
        # If there are queued messages, send next one directly
        next_message = conversation_queue.pop(0)
        response = {
            "threadId": thread_id,
            "onboarding": True,
            "message": next_message,
            "conversationQueue": conversation_queue  # return updated queue to frontend
        }
        print("[LOG] Sending queued message:", response)
        return jsonify(response)

    thread_id, assistant_reply = create_or_continue_thread(thread_id, user_message)
    print("[LOG] Daisy Assistant reply:", assistant_reply)

    onboarding_complete = "Onboarding complete." in assistant_reply
    print("[LOG] Onboarding complete status:", onboarding_complete)

    if onboarding_complete:
        search_query = None
        if "Onboarding complete. Here's your search query:" in assistant_reply:
            search_query = assistant_reply.split("Onboarding complete. Here's your search query:")[1].strip()
            print("[LOG] Extracted search query:", search_query)

        if search_query:
            celebs, keywords = parse_search_query(search_query)

            images_response = requests.get(
                f"{GOOGLE_AGENT_URL}/search-google-images",
                params={
                    "celebs": ",".join(celebs),
                    "keywords": ",".join(keywords)
                }
            ).json()

            images = images_response.get('images', [])
            filtered_images = filter_images(images, search_query)

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

    # Properly split assistant reply into individual messages/questions clearly:
    split_messages = [msg.strip() for msg in assistant_reply.strip().split("\n\n") if msg.strip()]
    next_message = split_messages.pop(0)

    response = {
        "threadId": thread_id,
        "onboarding": True,
        "message": next_message,
        "conversationQueue": split_messages  # queue remaining questions
    }
    print("[LOG] Single-question onboarding response with queue:", response)
    return jsonify(response)

def parse_search_query(search_query):
    known_celebrities = ['Hailey Bieber', 'Kendall Jenner', 'Jacob Elordi', 'Paul Mescal']
    celebs = []
    keywords = []

    for celeb in known_celebrities:
        if celeb.lower() in search_query.lower():
            celebs.append(celeb)
            search_query = search_query.lower().replace(celeb.lower(), "").strip()

    keywords = search_query.split()
    return celebs, keywords



if __name__ == '__main__':
    print("[LOG] Starting Daisy backend...")
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 8000)))
