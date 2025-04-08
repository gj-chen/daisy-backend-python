import os
import uuid
import json
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI
import psycopg2
from psycopg2.extras import RealDictCursor

# Load environment variables
load_dotenv()

# Flask app
app = Flask(__name__)
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})

# Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ASSISTANT_ID = os.getenv("DAISY_ASSISTANT_ID")
GOOGLE_IMAGE_AGENT_URL = os.getenv("GOOGLE_IMAGE_AGENT_URL")

# Thread tracking
THREADS = {}

# Home ping
@app.route('/')
def home():
    return "✅ Daisy backend is running — assistant-powered."

# Database connection
def get_pg_connection():
    return psycopg2.connect(
        host=os.getenv("SUPABASE_DB_HOST"),
        dbname=os.getenv("SUPABASE_DB_NAME"),
        user=os.getenv("SUPABASE_DB_USER"),
        password=os.getenv("SUPABASE_DB_PASSWORD"),
        cursor_factory=RealDictCursor
    )

def get_embedding_from_text(text: str):
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        raise RuntimeError(f"OpenAI embedding failed: {str(e)}")

# ✅ POST /chat
@app.route("/chat", methods=["POST"])
def chat_with_daisy():
    data = request.get_json()
    user_id = data.get("userId", "default")
    user_message = data.get("message")

    if not user_message:
        return jsonify({"error": "Missing message field"}), 400

    print(f"[LOG] Message from '{user_id}': {user_message}")

    # Use or create thread
    thread_id = THREADS.get(user_id)
    if thread_id:
        runs = client.beta.threads.runs.list(thread_id=thread_id, limit=1)
        if runs.data and runs.data[0].status in ["queued", "in_progress", "requires_action"]:
            return jsonify({"error": "Daisy is still thinking. Please wait."}), 429
        print(f"[LOG] Using existing thread: {thread_id}")
    else:
        thread = client.beta.threads.create()
        thread_id = thread.id
        THREADS[user_id] = thread_id
        print(f"[LOG] Created new thread: {thread_id}")

    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=user_message
    )
    print("[LOG] Added message to thread.")

    run = client.beta.threads.runs.create(
        assistant_id=ASSISTANT_ID,
        thread_id=thread_id
    )
    print(f"[LOG] Started run: {run.id}")

    tool_output = None

    while True:
        run_status = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
        if run_status.status == "completed":
            print("[LOG] Assistant run completed.")
            break
        elif run_status.status == "requires_action":
            print("[LOG] Tool call detected.")
            tool_calls = run_status.required_action.submit_tool_outputs.tool_calls
            tool_outputs = []

            for tool in tool_calls:
                if tool.function.name in ["search_pinterest", "search_curated_images"]:
                    args = json.loads(tool.function.arguments)
                    query = args.get("query", "")
                    print(f"[LOG] Daisy search query: {query}")

                    try:
                        embedding = get_embedding_from_text(query)
                        with get_pg_connection() as conn:
                            with conn.cursor() as cur:
                                cur.execute("""
                                    SELECT image_url
                                    FROM moodboard_items
                                    WHERE embedding IS NOT NULL
                                    ORDER BY embedding <=> %s::vector
                                    LIMIT 12;
                                """, (embedding,))
                                rows = cur.fetchall()
                                image_urls = [row["image_url"] for row in rows]
                                tool_output = { "imageUrls": image_urls }
                                print(f"[LOG] Found {len(image_urls)} images via semantic search.")
                    except Exception as e:
                        tool_output = {"imageUrls": []}
                        print("[ERROR] Semantic search failed:", e)

                    tool_outputs.append({
                        "tool_call_id": tool.id,
                        "output": json.dumps(tool_output)
                    })

            client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread_id,
                run_id=run.id,
                tool_outputs=tool_outputs
            )
            print("[LOG] Submitted tool outputs.")

        elif run_status.status in {"cancelled", "failed", "expired"}:
            print(f"[ERROR] Run {run_status.status}")
            return jsonify({"error": f"Run {run_status.status}"}), 500


    

    messages = client.beta.threads.messages.list(thread_id=thread_id)
    final_message = messages.data[0]
    reply_text = final_message.content[0].text.value
    print("[LOG] Daisy's reply:", reply_text)

    response = {
        "threadId": thread_id,
        "message": reply_text
    }

    if tool_output:
        response["moodboard"] = {
            "imageUrls": tool_output.get("imageUrls", []),
            "rationale": tool_output.get("rationale", {})
        }

    return jsonify(response)

# ✅ POST /search-images
@app.route("/search-images", methods=["POST"])
def search_images():
    try:
        data = request.get_json()
        summary = data.get("summary", "").strip()

        if not summary:
            return jsonify({"error": "Missing style summary"}), 400

        embedding = get_embedding_from_text(summary)

        with get_pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, image_url, source_url, metadata, 
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM moodboard_items
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT 12;
                """, (embedding, embedding))

                results = cur.fetchall()
                return jsonify({"images": results}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ POST /final-moodboard
@app.route("/final-moodboard", methods=["POST"])
def generate_final_moodboard():
    try:
        data = request.get_json()
        image_urls = data.get("imageUrls", [])
        include_guide = data.get("includeStylingGuide", False)

        if not image_urls:
            return jsonify({"error": "No image URLs provided"}), 400

        user_prompt = f"""
The user has selected the following images for their final moodboard:

{chr(10).join(image_urls)}

Please generate a final stylist summary that explains the overall vibe and aesthetic direction these images represent.

{"Also include a short styling guide explaining why each item works and how to wear it." if include_guide else "Keep it concise and visual without a styling guide."}
"""

        thread = client.beta.threads.create()
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_prompt.strip()
        )

        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=ASSISTANT_ID
        )

        while True:
            run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            if run_status.status == "completed":
                break
            elif run_status.status in ["failed", "cancelled", "expired"]:
                return jsonify({"error": f"Run failed: {run_status.status}"}), 500

        messages = client.beta.threads.messages.list(thread_id=thread.id)
        final_message = messages.data[0].content[0].text.value.strip()

        return jsonify({ "response": final_message }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ POST /upload-image
@app.route('/upload-image', methods=['POST'])
def upload_image():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file provided"}), 400

    file_extension = file.filename.split('.')[-1]
    file_name = f"{uuid.uuid4()}.{file_extension}"
    response = supabase.storage.from_('moodboard-images').upload(file_name, file.read())

    if response.status_code != 200:
        return jsonify({"error": "Upload failed", "details": response.json()}), 500

    public_url = f"{SUPABASE_URL}/storage/v1/object/public/moodboard-images/{file_name}"
    return jsonify({"image_url": public_url})

# Run server
if __name__ == '__main__':
    print("[LOG] Starting Daisy backend on port 5000...")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
