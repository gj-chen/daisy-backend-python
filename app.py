import os
import uuid
import json
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


# Thread tracking
THREADS = {}
STYLE_SUMMARIES = {}  # maps user_id → last style_summary
LAST_REPLY = {}       # tracks last assistant reply per user

# Database connection
def get_pg_connection():
    return psycopg2.connect(
        host=os.getenv("SUPABASE_DB_HOST"),
        dbname=os.getenv("SUPABASE_DB_NAME"),
        user=os.getenv("SUPABASE_DB_USER"),
        password=os.getenv("SUPABASE_DB_PASSWORD"),
        cursor_factory=RealDictCursor
    )


@app.route('/')
def home():
    return "✅ Daisy backend is running — assistant-powered."


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

@app.route("/chat", methods=["POST"])
def chat_with_daisy():
    data = request.get_json()
    user_id = data.get("userId", "default")
    user_message = data.get("message")
    print(f"[LOG] Message from '{user_id}': {user_message}")

    thread_id = THREADS.get(user_id)
    if not thread_id:
        thread = client.beta.threads.create()
        thread_id = thread.id
        THREADS[user_id] = thread_id
    client.beta.threads.messages.create(thread_id=thread_id, role="user", content=user_message)
    run = client.beta.threads.runs.create(thread_id=thread_id, assistant_id=ASSISTANT_ID)

    while run.status in ["queued", "in_progress"]:
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)

    tool_outputs = []
    frontend_images = []
    image_rationales = []
    tool_used = False

    if run.status == "requires_action" and run.required_action.type == "submit_tool_outputs":
        for tool_call in run.required_action.submit_tool_outputs.tool_calls:
            args = json.loads(tool_call.function.arguments)
            if tool_call.function.name == "search_curated_images":
                tool_used = True
                query = args.get("query")
                print(f"[TOOL] Daisy tool call — search_curated_images: {query}")
                STYLE_SUMMARIES[user_id] = query
                try:
                    results = search_images_from_db(query, user_id)
                    frontend_images = results
                    image_rationales = [{"url": img["url"], "rationale": img["explanation"]} for img in results]
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": json.dumps({ "images": results or [] })
                    })
                except Exception as e:
                    print(f"[ERROR] image search failed: {e}")
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": json.dumps({ "images": [] })
                    })

        run = client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread_id, run_id=run.id, tool_outputs=tool_outputs
        )

        while run.status in ["queued", "in_progress", "requires_action"]:
            run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)

    # Final reply logic
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    
    assistant_reply = ""
    for m in reversed(messages.data):
        if m.role == "assistant" and m.content:
            raw = m.content[0].text.value.strip()
            lines = raw.splitlines()
            clean = [line for line in lines if "http" not in line and not line.startswith("-")]
            assistant_reply = "\n".join(clean).strip()

            # Check for repetition
            last_used = LAST_REPLY.get(user_id, "")
            if assistant_reply == last_used:
                assistant_reply += "\nLet me take a fresh look at this with you."
            break

    LAST_REPLY[user_id] = assistant_reply

    def split_message(text, max_len=280):
        sentences = re.split(r'(?<=[.!?]) +', text.strip())
        chunks, chunk = [], ""
        for s in sentences:
            if len(chunk) + len(s) <= max_len:
                chunk += " " + s if chunk else s
            else:
                chunks.append(chunk.strip())
                chunk = s
        if chunk:
            chunks.append(chunk.strip())
        return chunks

    return jsonify({
        "messages": [{"role": "assistant", "text": t} for t in split_message(assistant_reply)],
        "threadId": thread_id,
        "moodboard": { "images": image_rationales },
        "toolUsed": tool_used  # ✅ now included
    })

def search_images_from_db(query, user_id):
    embedding = get_embedding_from_text(query)
    STYLE_SUMMARIES[user_id] = query

    conn = get_pg_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, stored_image_url, source_url
        FROM moodboard_items
        WHERE embedding IS NOT NULL
        ORDER BY embedding <-> %s::vector
        LIMIT 10;
    """, (embedding,))
    rows = cur.fetchall()
    conn.close()

    images = [{"id": r["id"], "url": r["stored_image_url"], "source_url": r.get("source_url", "")} for r in rows]
    image_block = "\n".join(f"{i+1}. {img['url']}" for i, img in enumerate(images))

    prompt = f"""
    You are a fashion stylist. The user described their style direction as:
    "{query}"

    You've selected the following fashion images. For each, write a 1–2 sentence stylist explanation in a casual, confident tone. 
    Speak as a stylist would — no catalogs, no formality. Think: why it works, how to wear it, what vibe it hits.

    Images:
    {image_block}

    Respond as JSON:
    [{{ "url": "...", "explanation": "..." }}, ...]
    Only return JSON.
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a fashion stylist writing image rationales."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    return json.loads(response.choices[0].message.content)


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
        user_id = data.get("userId", "default")
        style_summary = STYLE_SUMMARIES.get(user_id, "No specific styling summary provided.")

        if not image_urls:
            return jsonify({"error": "No image URLs provided"}), 400

        # Pull metadata for each selected image
        conn = get_pg_connection()
        cursor = conn.cursor()

        placeholders = ','.join(['%s'] * len(image_urls))
        sql = f"""
            SELECT stored_image_url, metadata
            FROM moodboard_items
            WHERE stored_image_url IN ({placeholders});
        """
        cursor.execute(sql, tuple(image_urls))
        results = cursor.fetchall()
        conn.close()

        image_metadata = [
            {
                "url": row["stored_image_url"],
                "metadata": row["metadata"]
            }
            for row in results
        ]

        # Generate the stylist prompt
        user_prompt = f"""
The user has selected the following fashion images for their final moodboard.
Each image includes styling metadata (fit, body suitability, fabric, season, occasion, etc).

Their overall styling context is:
"{style_summary}"

Please:
1. Write a 2–3 sentence summary that captures the overall aesthetic direction of this moodboard.
2. Write a short rationale for each image explaining why it works and how it supports the moodboard direction.

Images:
{json.dumps(image_metadata, indent=2)}

Respond ONLY as valid JSON like this:
{{
  "summary": "...",
  "rationales": ["...", "..."]
}}
        """.strip()

        # Call Daisy
        thread = client.beta.threads.create()
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_prompt
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
        gpt_response = messages.data[0].content[0].text.value.strip()

        parsed = json.loads(gpt_response)
        return jsonify(parsed), 200

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
