import os
from dotenv import load_dotenv
from google import genai
from .llm_prompts import spell_checker, rewriter, expander

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)

def llm_spell_check(query):
    response = client.models.generate_content(
        model="gemma-3-27b-it",
        contents=spell_checker.format(query=query),
    )

    corrected = (response.text or "").strip().strip('"')
    return corrected if corrected else query

def llm_rewrite(query):
    response = client.models.generate_content(
        model="gemma-3-27b-it",
        contents=rewriter.format(query=query),
    )

    corrected = (response.text or "").strip().strip('"')
    return corrected if corrected else query

def llm_expand(query):
    response = client.models.generate_content(
        model="gemma-3-27b-it",
        contents=expander.format(query=query),
    )

    expansion = (response.text or "").strip().strip('"')
    if not expansion:
        return query
    return f"{query} {expansion}"