# routes/chat.py
import os
import requests
from flask import Blueprint, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()

router = Blueprint("chat", __name__)
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ASSISTANT_ID = os.getenv("DAISY_ASSISTANT_ID")
GOOGLE_IMAGE_AGENT_URL = os.environ['GOOGLE_IMAGE_AGENT_URL']

THREADS = {}

@router.route("/chat", methods=["POST"])
def chat_with_daisy():
    data = request.get_json()
    user_id = data.get("userId", "default")
    user_message = data.get("message")

    if not user_message:
        return jsonify({"error": "Missing message field"}), 400

    print(f"[LOG] Message from '{user_id}': {user_message}")

    if user_id in THREADS:
        thread_id = THREADS[user_id]
        # Check if any runs are still in progress
        runs = openai.beta.threads.runs.list(thread_id=thread_id, limit=1)
        if runs.data and runs.data[0].status in ["queued", "in_progress", "requires_action"]:
            return jsonify({"error": "Daisy is still thinking. Please wait."}), 429

    thread_id = THREADS.get(user_id)
    if not thread_id:
        thread = openai.beta.threads.create()
        thread_id = thread.id
        THREADS[user_id] = thread_id
        print(f"[LOG] Created new thread: {thread_id}")
    else:
        print(f"[LOG] Using thread: {thread_id}")

    openai.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=user_message
    )
    print("[LOG] Added message to thread.")

    run = openai.beta.threads.runs.create(
        assistant_id=ASSISTANT_ID,
        thread_id=thread_id
    )
    print(f"[LOG] Started run: {run.id}")

    tool_output = None
    while True:
        run_status = openai.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
        if run_status.status == "completed":
            print("[LOG] Assistant run completed.")
            break
        elif run_status.status == "requires_action":
            print("[LOG] Tool call detected.")
            tool_calls = run_status.required_action.submit_tool_outputs.tool_calls
            tool_outputs = []

            for tool in tool_calls:
                if tool.function.name == "search_pinterest":  # Actually calling Google Images agent
                    
                    args = json.loads(tool.function.arguments)
                    query = args.get("query", "")
                    print(f"[LOG] Daisy query: {query}")
                    try:
                        r = requests.get(GOOGLE_IMAGE_AGENT_URL, params={"q": query}, timeout=10)
                        r.raise_for_status()
                        tool_output = r.json()
                        print(f"[LOG] Retrieved {len(tool_output.get('imageUrls', []))} images.")
                    except Exception as e:
                        tool_output = {"imageUrls": [], "rationale": {}}
                        print("[ERROR] Image agent failed:", e)

                    tool_outputs.append({
                        "tool_call_id": tool.id,
                        "output": json.dumps(tool_output)
                    })

            openai.beta.threads.runs.submit_tool_outputs(
                thread_id=thread_id,
                run_id=run.id,
                tool_outputs=tool_outputs
            )
            print("[LOG] Submitted tool outputs.")
        elif run_status.status in {"cancelled", "failed", "expired"}:
            print(f"[ERROR] Run {run_status.status}")
            return jsonify({"error": f"Run {run_status.status}"}), 500

    messages = openai.beta.threads.messages.list(thread_id=thread_id)
    final_message = messages.data[0]

    reply_text = final_message.content[0].text.value
    print("[LOG] Daisy's final message:", reply_text)

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
