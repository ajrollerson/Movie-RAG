import os
import json
from dotenv import load_dotenv
from google import genai
from .llm_prompts import spell_checker, rewriter, expander, rerank, batch, evaluate

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)

def results_to_doc_list_str(results):
    lines = []
    for result in results:
        lines.append(f"ID: {result['id']} | Title: {result['title']} | Description: {result['description']}")

    doc_list_str = "\n".join(lines)
    return doc_list_str 

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

def llm_rerank(query, result):
    response = client.models.generate_content(
        model="gemma-3-27b-it",
        contents=rerank.format(
            query=query,
            title=result["title"],
            description=result["description"]
            ),
    )
    text = (response.text or "").strip().strip('"')
    try:
        score = float(text) 
    except ValueError:
        score = 0          
    return score

def llm_batch(query, results):
    response = client.models.generate_content(
        model="gemma-3-27b-it",
        contents=batch.format(
            query=query,
            doc_list_str=results_to_doc_list_str(results)
            ),
    )

    text = response.text.strip().strip("`").strip()
    if text.startswith("json"):
        text = text[4:].strip()
    json_data = json.loads(text)
    return json_data

def llm_evaluate(query, results):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=evaluate.format(
            query=query,
            formatted_results=chr(10).join(results_to_doc_list_str(results))
            ),
    )

    text = response.text.strip().strip("`").strip()
    if text.startswith("json"):
        text = text[4:].strip()
    json_data = json.loads(text)
    return json_data

