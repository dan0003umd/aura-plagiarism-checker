# gemini_utils.py
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env file")

genai.configure(api_key=API_KEY)
DEFAULT_MODEL = "models/text-bison-001"

def generate_rewrite(
    prompt: str, 
    model: str = DEFAULT_MODEL, 
    temperature: float = 0.35,
    max_output_tokens: int = 400
):
    """
    Uses Gemini to generate an original rewrite of the given text.
    Gracefully falls back to available methods across SDK versions.
    """
    try:
        if hasattr(genai, "TextGenerationClient"):
            from google.generativeai.client import TextGenerationClient
            client = TextGenerationClient(api_key=API_KEY)
            response = client.generate(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens
            )
            if hasattr(response, "output") and len(response.output) > 0:
                return response.output[0].content.strip()
            return str(response)
        elif hasattr(genai, "chat"):
            response = genai.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            if "candidates" in response and response["candidates"]:
                return response["candidates"][0].get("content", "").strip()
            return str(response)
        elif hasattr(genai, "generate_text"):
            response = genai.generate_text(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens
            )
            if hasattr(response, "text"):
                return response.text.strip()
            if isinstance(response, dict) and "candidates" in response:
                return response["candidates"][0].get("content", "").strip()
            return str(response)
        else:
            return prompt
    except Exception as e:
        return f"(Gemini error) {e}"

