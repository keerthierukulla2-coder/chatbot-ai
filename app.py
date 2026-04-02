from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google import genai
from google.genai import types
import os

load_dotenv()

app = Flask(__name__)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is missing. Add it to your .env file.")

client = genai.Client(api_key=api_key)

MODEL_NAME = "gemini-2.5-flash"

SYSTEM_INSTRUCTION = """
You are an intelligent conversational AI assistant.

Rules:
1. Answer clearly and naturally.
2. Be concise unless the user asks for more detail.
3. For current or time-sensitive topics, use grounded web search.
4. If a fact may be uncertain, say so instead of guessing.
5. Be helpful, polite, and accurate.
"""

def extract_sources(response):
    sources = []
    seen = set()

    try:
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            grounding = getattr(candidate, "grounding_metadata", None)
            if not grounding:
                continue

            chunks = getattr(grounding, "grounding_chunks", None) or []
            for chunk in chunks:
                web_info = getattr(chunk, "web", None)
                if not web_info:
                    continue

                title = getattr(web_info, "title", None)
                url = getattr(web_info, "uri", None)

                if url and url not in seen:
                    seen.add(url)
                    sources.append({
                        "title": title or "Source",
                        "url": url
                    })
    except Exception:
        pass

    return sources


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()

    if not user_message:
        return jsonify({
            "response": "Please type a message.",
            "sources": []
        }), 400

    try:
        grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            tools=[grounding_tool],
            temperature=0.4,
            max_output_tokens=500,
        )

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_message,
            config=config,
        )

        reply = getattr(response, "text", None) or "I could not generate an answer."
        sources = extract_sources(response)

        return jsonify({
            "response": reply,
            "sources": sources
        })

    except Exception as e:
        return jsonify({
            "response": f"Error: {str(e)}",
            "sources": []
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
