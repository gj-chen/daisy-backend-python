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
import re

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
STYLE_SUMMARIES = {}  # maps user_id → last style_summary


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


def refine_style_summary(previous_summary, user_feedback):
    prompt = f"""
You are a fashion stylist working with the following moodboard direction:

"{previous_summary}"

The client just said:
"{user_feedback}"

Please generate a revised 1–2 sentence styling summary that adjusts the direction accordingly. Be intuitive and taste-driven. Keep it short and emotional.

Only return the updated summary.
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            { "role": "system", "content": "You are a fashion stylist refining a vibe summary." },
            { "role": "user", "content": prompt }
        ],
        temperature=0.7
    )

    return response.choices[0].message.content.strip()



@app.route('/test-db')
def test_db():
    try:
        conn = get_pg_connection()
        conn.close()
        return "✅ DB connection successful!"
    except Exception as e:
        return f"❌ DB error: {str(e)}"


# ✅ POST /chat
def split_message(text, max_group_len=280):
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_group_len:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

@app.route("/chat", methods=["POST"])
def chat_with_daisy():
    print(f"[DEBUG] Using Assistant ID: {ASSISTANT_ID}")
    data = request.get_json()
    user_id = data.get("userId", "default")
    user_message = data.get("message")

    if not user_message:
        return jsonify({"error": "Missing message field"}), 400

    print(f"[LOG] Message from '{user_id}': {user_message}")

    thread_id = THREADS.get(user_id)
    if not thread_id:
        thread = client.beta.threads.create()
        thread_id = thread.id
        THREADS[user_id] = thread_id
        print(f"[LOG] Created new thread: {thread_id}")
    else:
        print(f"[LOG] Using thread: {thread_id}")

    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=user_message
    )

    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=ASSISTANT_ID,
    )

    while run.status in ["queued", "in_progress"]:
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)

    print(f"[RUN STATUS] {run.status}")

    # Log tool calls if required
    if run.status == "requires_action":
        print("[REQUIRES ACTION]")
        if run.required_action.type == "submit_tool_outputs":
            tool_calls = run.required_action.submit_tool_outputs.tool_calls
            print(f"[TOOL CALLS RECEIVED]: {len(tool_calls)}")
            for tool_call in tool_calls:
                print(f"[TOOL REQUESTED]: {tool_call.function.name} — Args: {tool_call.function.arguments}")
        else:
            print("[REQUIRES ACTION] But not a tool call.")

    tool_outputs = []

    if run.status == "requires_action" and run.required_action.type == "submit_tool_outputs":
        for tool_call in run.required_action.submit_tool_outputs.tool_calls:
            tool_call_id = tool_call.id
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            if function_name == "search_curated_images":
                query = arguments.get("query")
                print(f"[TOOL] Daisy wants to search curated images for: {query}")

                # Track original style summary
                STYLE_SUMMARIES[user_id] = query

                try:
                    image_results = search_images_from_db(query, user_id)
                    tool_outputs.append({
                        "tool_call_id": tool_call_id,
                        "output": json.dumps({ "images": image_results or [] })
                    })
                except Exception as e:
                    print(f"[ERROR] Failed to fetch images: {e}")
                    tool_outputs.append({
                        "tool_call_id": tool_call_id,
                        "output": json.dumps({ "images": [] })
                    })

        run = client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread_id,
            run_id=run.id,
            tool_outputs=tool_outputs
        )

        while run.status in ["queued", "in_progress"]:
            run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)

    # Final message parsing
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    full_text = ""
    for m in messages.data:
        if m.role == "assistant" and m.content:
            full_text = m.content[0].text.value.strip()
            break

    print(f"[LOG] Daisy's full reply: {full_text}")

    chunks = split_message(full_text)

    moodboard_images = []
    if tool_outputs:
        try:
            moodboard_images = json.loads(tool_outputs[-1]["output"])["images"]
        except Exception as e:
            print(f"[ERROR] Failed to extract images for frontend: {e}")

    return jsonify({
        "messages": [{"role": "assistant", "text": chunk} for chunk in chunks],
        "threadId": thread_id,
        "moodboard": {
            "images": moodboard_images
        }
    })



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

def search_images_from_db(query, user_id):
    embedding = get_embedding_from_text(query)
    
    STYLE_SUMMARIES[user_id] = query
    
    conn = get_pg_connection()
    cursor = conn.cursor()

    sql = """
    SELECT id, stored_image_url, title, source_url
    FROM moodboard_items
    WHERE embedding IS NOT NULL
    ORDER BY embedding <-> %s::vector
    LIMIT 6;
    """
    cursor.execute(sql, (embedding,))
    results = cursor.fetchall()
    conn.close()

    image_data = [
        {
            "id": row["id"],
            "url": row["stored_image_url"],
            "title": row.get("title", "Untitled"),
            "source_url": row.get("source_url", "")
        }
        for row in results
    ]

    # Build prompt to send to GPT-4 for rationale generation
    prompt = f"""
You are a fashion stylist. Your client described their style direction as:
"{query}"

You have selected {len(image_data)} fashion images to illustrate this direction. For each image, write 1–2 short sentences explaining *why* it fits the styling goal above.

Images:
{chr(10).join(f"{i+1}. {img['title']} – {img['url']}" for i, img in enumerate(image_data))}

Respond as a JSON array of objects like:
[
  {{ "url": "...", "explanation": "..." }},
  ...
]
Only return the JSON.
""".strip()

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a fashion stylist writing brief rationales for a curated moodboard."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    # Parse the assistant's response
    explanations = json.loads(response.choices[0].message.content)

    return explanations




# Run server
if __name__ == '__main__':
    print("[LOG] Starting Daisy backend on port 5000...")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
