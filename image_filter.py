import os
from openai import OpenAI
from dotenv import load_dotenv
import json
import re

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def filter_images(images, description):
    prompt = f"""
    You're Daisy—a visual identity expert, stylist, and creative director. You have detailed context about the user:

    {description}

    Given this context, select the 5 most visually aligned images from this set:

    {images}

    Prioritize:
    - Style consistency (matches user's desired vibe and aesthetic)
    - Authenticity (reflects real, wearable looks inspired by the mentioned celebrity or archetype)
    - Relevance to user's body shape and occasion

    Return ONLY a JSON array of URLs: ["url1", "url2", ...]
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You’re Daisy, a highly intuitive visual style expert who precisely matches user profiles with ideal fashion inspiration."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0
    )

    raw_content = response.choices[0].message.content.strip()
    print("[LOG] GPT-4 Vision raw response:", raw_content)

    json_match = re.search(r'\[.*\]', raw_content, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
    else:
        print("[ERROR] No JSON array found in GPT response.")
        return []

    try:
        urls = json.loads(json_str)
        print("[LOG] Successfully parsed JSON:", urls)
        return urls
    except json.JSONDecodeError as e:
        print("[ERROR] JSON decode error after extraction:", e)
        return []
