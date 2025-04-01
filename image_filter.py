import os
from openai import OpenAI
from dotenv import load_dotenv
import json
import re

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def filter_images(images, description):
    prompt = f"""
        You're Daisy—a visual identity expert and stylist.
    
        Given the user's described vibe and aesthetic:
    
        {description}
    
        Select the 8 best-aligned images from these candidates:
    
        {images}
    
        ✅ Prioritize images that:
        - Clearly match the user's requested style and aesthetic.
        - Have natural lighting, realistic environments, and high visual quality.
        - Feature full-body or half-body outfits, visibly styled.
    
        🚫 Exclude images that:
        - Contain any text, prices, overlays, watermarks, or logos.
        - Are collages, editorial campaigns, ads, runways, or overly stylized/editorial poses.
        - Appear low-resolution, overly posed, or artificial.
    
        Return ONLY a JSON array of image URLs: ["url1", "url2", ...]
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
