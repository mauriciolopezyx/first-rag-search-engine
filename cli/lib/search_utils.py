import os
import argparse
from dotenv import load_dotenv
from google import genai
from google.genai import types
import json

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

def spell_check_query(query: str):
    messages = [types.Content(role="user", parts=[types.Part(text=query)])]
    res = client.models.generate_content(
        model="gemini-2.5-lite", 
        contents=messages,
        config=types.GenerateContentConfig(
            system_instruction=f"""
            Fix any spelling errors in the given user query.

            Only correct obvious typos. Don't change correctly spelled words.

            If no errors, return the original query.
            Corrected:
            """
        )
    )

    print(f"Prompt tokens: {res.usage_metadata.prompt_token_count}\nResponse tokens: {res.usage_metadata.candidates_token_count}")
    return res.text

def rewrite_query(query: str):
    messages = [types.Content(role="user", parts=[types.Part(text=query)])]
    res = client.models.generate_content(
        model="gemini-2.5-lite", 
        contents=messages,
        config=types.GenerateContentConfig(
            system_instruction=
            f"""Rewrite the given user query to be more specific and searchable.

            Consider:
            - Common movie knowledge (famous actors, popular films)
            - Genre conventions (horror = scary, animation = cartoon)
            - Keep it concise (under 10 words)
            - It should be a google style search query that's very specific
            - Don't use boolean logic

            Examples:

            - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
            - "movie about bear in london with marmalade" -> "Paddington London marmalade"
            - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

            Rewritten query:"""
        )
    )

    print(f"Prompt tokens: {res.usage_metadata.prompt_token_count}\nResponse tokens: {res.usage_metadata.candidates_token_count}")
    return res.text

def expand_query(query: str):
    messages = [types.Content(role="user", parts=[types.Part(text=query)])]
    res = client.models.generate_content(
        model="gemini-2.5-lite", 
        contents=messages,
        config=types.GenerateContentConfig(
            system_instruction=
            f"""Expand the given user query with related terms.

            Add synonyms and related concepts that might appear in movie descriptions.
            Keep expansions relevant and focused.
            This will be appended to the original query.

            Examples:

            - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
            - "action movie with bear" -> "action thriller bear chase fight adventure"
            - "comedy with bear" -> "comedy funny bear humor lighthearted"

            Expanded query:
            """
        )
    )

    print(f"Prompt tokens: {res.usage_metadata.prompt_token_count}\nResponse tokens: {res.usage_metadata.candidates_token_count}")
    return res.text

def batch_rerank_results(query: str, str_docs: list[str]) -> list[int]:
    doc_list_str = "\n---- NEW ENTRY ----\n".join(str_docs)

    messages = [types.Content(role="user", parts=[types.Part(text=query)])]
    res = client.models.generate_content(
        model="gemini-2.5-flash", 
        contents=messages,
        config=types.GenerateContentConfig(
            system_instruction=
            f"""Rank these movies by relevance to the given user query. The delimiter between new entries is "---- NEW ENTRY ----".

            Movies:
            {doc_list_str}

            Return ALL of the given IDs ONLY in order of relevance, with best matches first. Return a valid python list AS A STRING, nothing else.
            
            For example, if we were given a Movies string with 5 entries (hence 5 ids), one example could be:
            [75, 12, 34, 2, 1]
            As you can tell, the resulting list contains all 5 initial ids.
            
            """
        )
    )
    print("LLM batch response:", res.text)
    order = json.loads(res.text)
    print(f"Prompt tokens: {res.usage_metadata.prompt_token_count}\nResponse tokens: {res.usage_metadata.candidates_token_count}\nResults Order is: {order}")
    return order