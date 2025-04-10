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
import openai

# OpenAI setup
openai.api_key = os.getenv("OPENAI_API_KEY")


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


USER_CONTEXTS = {}  # user_id -> { last_user, last_daisy }
STYLE_SUMMARIES = {}  # user_id -> current style summary
THREADS = {}  # user_id -> OpenAI thread ID
LAST_MESSAGE_ID = {}  # user_id -> last assistant message ID
IMAGE_FEEDBACK = {}  # user_id -> { url: 'like' | 'dislike' }


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
    if not isinstance(text, str):
        print("[ERROR] split_message received non-string input:", type(text), repr(text))
        return ["Let’s see if any of these ideas spark something."]

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
    response = client.embeddings.create(input=text, model="text-embedding-ada-002")
    return response.data[0].embedding


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
        model="gpt-4-turbo",
        messages=[
            { "role": "system", "content": "You are a fashion stylist refining a vibe summary." },
            { "role": "user", "content": prompt }
        ],
        temperature=0.7
    )

    try:
        print("[DEBUG] Chat completion response:", response)
        return response.choices[0].message.content.strip()
    except (IndexError, AttributeError, KeyError, TypeError) as e:
        print(f"[ERROR] GPT call failed safely: {e}")
        return "Updated summary not available — please try again."

# Daisy system prompt (keep full version in actual implementation)
with open("./backend/prompts/daisy_assistant_prompt.txt", "r") as f:
    DAISY_SYSTEM_PROMPT = f.read()

# In-memory history (swap with Redis or DB for production)
CONVERSATION_HISTORY = {}

@app.route("/chat", methods=["POST"])
def chat_with_daisy():
    data = request.get_json()
    user_id = data.get("userId", "default")
    user_message = data.get("message")

    # Special case: user clicked "Refine based on feedback"
    if "refine based on feedback" in user_message.lower():
        print("[DEBUG] Triggering refinement from feedback")

        prev_summary = STYLE_SUMMARIES.get(user_id, "")
        feedback_map = IMAGE_FEEDBACK.get(user_id, {})

        if not feedback_map:
            return jsonify({"messages": [{"role": "assistant", "text": "I haven’t seen your feedback yet — like or dislike a few looks first!"}]}), 200

        liked = [k for k, v in feedback_map.items() if v == "like"]
        disliked = [k for k, v in feedback_map.items() if v == "dislike"]
        feedback_description = ""

        if liked:
            feedback_description += f"You liked these looks: {', '.join(liked[:3])}. "
        if disliked:
            feedback_description += f"You disliked these: {', '.join(disliked[:3])}. "

        refined_summary = refine_style_summary(prev_summary, feedback_description)
        STYLE_SUMMARIES[user_id] = refined_summary

        rationale_output = search_images_from_db(refined_summary, user_id)
        images = rationale_output
        tool_used = True

        followup_prompt = f"""
        You just revised the moodboard based on the user's feedback.

        Style summary: "{refined_summary}"

        Write a warm, stylish 1–2 sentence comment acknowledging the refinement — and inviting the user to explore the new ideas.
        """

        followup_response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are Daisy, a stylist refining based on feedback."},
                {"role": "user", "content": followup_prompt}
            ],
            temperature=0.7
        )

        print("[DEBUG] GPT raw response:", followup_response)

        try:
            assistant_reply = followup_response.choices[0].message.content.strip()
            print("[DEBUG] Extracted message:", assistant_reply)
        except (IndexError, AttributeError, KeyError, TypeError) as e:
            print(f"[ERROR] GPT call failed safely: {e}")
            assistant_reply = "Let’s see if any of these ideas spark something."

        history = CONVERSATION_HISTORY.get(user_id, [])
        history.append({"role": "assistant", "content": assistant_reply})
        CONVERSATION_HISTORY[user_id] = history

        print("[DEBUG] Final images list to return:", images)
        
        return jsonify({
            "messages": [{"role": "assistant", "text": assistant_reply}],
            "threadId": "n/a",
            "moodboard": {"images": images},
            "toolUsed": tool_used
        })

    print(f"[LOG] Message from '{user_id}': {user_message}")

    history = CONVERSATION_HISTORY.get(user_id, [])
    history.append({"role": "user", "content": user_message})

    scoped_history = history[-6:] if len(history) > 6 else history
    messages = [{"role": "system", "content": DAISY_SYSTEM_PROMPT}] + scoped_history

    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.8
        )

        print("[DEBUG] GPT raw response:", response)

        try:
            assistant_reply = response.choices[0].message.content.strip()
            print("[DEBUG] Extracted message:", assistant_reply)
        except (IndexError, AttributeError, KeyError) as e:
            print("[ERROR] Failed to extract assistant reply:", e)
            assistant_reply = "Let’s see if any of these ideas spark something."


        tool_used = False
        images = []

        if "[[STYLE_SEARCH]]" in assistant_reply:
            assistant_reply = assistant_reply.replace("[[STYLE_SEARCH]]", "").strip()

            confirm_phrases = ["i'm ready", "lets go", "show me", "cool"]
            full_context = " ".join([m["content"].lower() for m in scoped_history if m["role"] == "user"])
            if not any(p in full_context for p in confirm_phrases):
                assistant_reply = "Gorgeous — let’s make sure I have what I need. Can I ask a few more quick things before I pull ideas?"
                CONVERSATION_HISTORY[user_id] = history + [{"role": "assistant", "content": assistant_reply}]
                return jsonify({
                    "messages": [{"role": "assistant", "text": assistant_reply}],
                    "threadId": "n/a",
                    "toolUsed": False
                })

            summary_prompt = f"""
            The user said: "{user_message}"

            Daisy replied: "{assistant_reply}"

            Now summarize this into a 1–2 sentence styling direction for a fashion image search.
            Focus on tone, vibe, silhouette, and use descriptive styling language.
            """

            summary_response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a stylist summarizing style directions."},
                    {"role": "user", "content": summary_prompt}
                ],
                temperature=0.7
            )

            print("[DEBUG] GPT raw response:", summary_response)

            try:
                style_summary = summary_response.choices[0].message.content.strip()
                print("[DEBUG] Extracted message:", style_summary)
            except (IndexError, AttributeError, KeyError, TypeError) as e:
                print(f"[ERROR] GPT call failed safely: {e}")
                style_summary = "Minimal creative polish — relaxed layers, artsy edge, nothing forced."

            STYLE_SUMMARIES[user_id] = style_summary

            rationale_output = search_images_from_db(style_summary, user_id)
            images = rationale_output
            tool_used = True

            followup_prompt = f"""
            You just showed the user a moodboard with these looks:

            "{style_summary}"

            Write a short 1–2 sentence stylist comment that welcomes the user into the board — minimal, stylish, and reflective.
            DO NOT describe individual items. That will be shown elsewhere.
            """

            followup_response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are Daisy, summarizing a moodboard you just curated."},
                    {"role": "user", "content": followup_prompt}
                ],
                temperature=0.7
            )


            try:
                assistant_reply = followup_response.choices[0].message.content.strip()
            except (IndexError, AttributeError, KeyError, TypeError) as e:
                print(f"[ERROR] GPT call failed safely: {e}")
                assistant_reply = "Let’s see if any of these ideas spark something."

        history.append({"role": "assistant", "content": assistant_reply})
        CONVERSATION_HISTORY[user_id] = history

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

        for img in images:
            if "explanation" in img:
                img["rationale"] = img.pop("explanation")

        # ✅ Fallback guard in case assistant_reply was never defined
        if "assistant_reply" not in locals():
            print("[ERROR] assistant_reply was never defined")
            assistant_reply = "Let’s see if any of these ideas spark something."

        print("[DEBUG] Type of assistant_reply:", type(assistant_reply))
        print("[DEBUG] Value of assistant_reply:", assistant_reply)

        # ✅ Safely split assistant reply and ensure all outputs are valid strings
        messages = []
        for t in split_message(assistant_reply):
            if isinstance(t, str):
                messages.append({"role": "assistant", "text": t})
            else:
                print("[ERROR] Invalid text value from split_message:", repr(t))
                messages.append({"role": "assistant", "text": "Let’s keep going — I’ll refine this idea."})

        # ✅ Safe return block
        return jsonify({
            "messages": messages,
            "threadId": "n/a",
            "moodboard": {"images": images},
            "toolUsed": tool_used
        })


    except Exception as e:
        print("[ERROR] OpenAI call failed:", e)
        return jsonify({"error": str(e)}), 500


def search_images_from_db(query, user_id):
    embedding = get_embedding_from_text(query)
    STYLE_SUMMARIES[user_id] = query

    conn = get_pg_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, stored_image_url, source_url, metadata
        FROM moodboard_items
        WHERE embedding IS NOT NULL
          -- AND embedding <-> %s::vector < 1.0
        ORDER BY embedding <-> %s::vector
        LIMIT 10;
    """, (embedding, embedding))  # Pass the embedding parameter twice
    rows = cur.fetchall()
    print("[DEBUG] Retrieved rows from DB:", rows)
    conn.close()

    results = []

    for row in rows:
        metadata = row.get("metadata")
        url = row["stored_image_url"]
        rationale = "This look was selected for its fit with the current style direction."

        # Only generate rationale if metadata exists
        if metadata:
            rationale_prompt = f"""

            You are Daisy, a stylist curating visuals for a client. They’re looking for a vibe that matches this direction: "{query}"

You’re reviewing this image and its metadata. Write a short, intuitive comment (1–2 sentences) explaining why this image fits. Be stylish and editorial — not formal, literal, or overly descriptive.

Use emotional tone. Never list fabric, pattern, or season unless it's essential to the vibe.

Only return your comment.

            If the metadata doesn’t seem relevant or the image feels off, keep the rationale very short and honest.

            Metadata:
            {json.dumps(metadata, indent=2)}

            Only return your rationale. No headers, no formatting.
            """
            try:
                response = openai.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are Daisy, a fashion stylist generating rationales."},
                        {"role": "user", "content": rationale_prompt}
                    ],
                    temperature=0.7
                )

                try:
                    rationale = response.choices[0].message.content.strip()
                except (IndexError, AttributeError, KeyError, TypeError) as e:
                    print(f"[ERROR] GPT call failed safely: {e}")
                    rationale = "Selected for visual alignment with the style summary."
            except Exception as e:
                print(f"[ERROR] Rationale generation failed for image {row['id']}: {e}")
                rationale = "Selected for visual alignment with the style summary."


        results.append({
            "id": row["id"],
            "url": url,
            "source_url": row.get("source_url", ""),
            "rationale": rationale
        })

    return results

@app.route("/feedback", methods=["POST"])
def record_image_feedback():
    data = request.get_json()
    user_id = data.get("userId", "default")
    url = data.get("imageUrl")
    value = data.get("value")  # "like" or "dislike"

    if user_id not in IMAGE_FEEDBACK:
        IMAGE_FEEDBACK[user_id] = {}

    IMAGE_FEEDBACK[user_id][url] = value
    print(f"[FEEDBACK] {user_id} → {value} on {url}")
    return jsonify({"status": "ok"})


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

        try:
            gpt_response = messages.data[0].content[0].text.value.strip()
        except (IndexError, AttributeError, KeyError, TypeError) as e:
            print(f"[ERROR] Final moodboard message parse failed: {e}")
            return jsonify({"error": "Failed to generate final moodboard summary."}), 500

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