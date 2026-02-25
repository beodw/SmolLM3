import os
import re
import json
import torch
import runpod
from transformers import pipeline, AutoTokenizer

# --- INITIALIZATION ---
# Using SmolLM3-3B with bfloat16 for speed and memory efficiency
model_id = "HuggingFaceTB/SmolLM3-3B"
device = 0 if torch.cuda.is_available() else -1

tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline(
    "text-generation", 
    model=model_id, 
    tokenizer=tokenizer, 
    device=device,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
)

def handler(job):
    job_input = job["input"]
    user_prompt = job_input.get("prompt", "A smooth r&b song")
    max_retries = 3
    attempts = 0

    messages = [
        {"role": "system", "content": "/no_think.You are a rigid API response generator. Output ONLY a raw JSON object. Do not include introductory text, explanations, or markdown code blocks."},
        {"role": "user", "content":f"""
        
        Your task is to transform a user's creative idea into 3 specific components: 
              1. A title. This is just 1 - 2 word string that is relevant to the user's song idea.
              2. A set of tags which are a single line of natural language tags. Include genre, mood, instrumentation, and tempo.
              3. A structured set of lyrics which include the following strict rules:
                  1. Use square brackets for sections: [intro-medium], [verse], [chorus], [inst-short].
                  2. Each line in the lyrics must be short, only a few syllables, followed by '...'. EVERY LINE OF LYRICS YOU WRITE MUST HAVE '...' before the new line character \n !
                  3. Each line should sound like song lyrics relevant to the user's idea. Do not put song descriptions here but actual lyrics that can be sang.
                  4. No punctuation.
                  5. Each line in the lyrics MUST rhyme with another one
                  

        Format your response like this:
            Example Format:
            {{
                "title": "My Ex Lover",
                "tags": "male voice, pop, melancholic, 90 BPM",
                "lyrics": "[intro-short]\noohhhhh...\n[verse]\nBaby, I miss you...\n[chorus]\nPlease come back...\n[verse]\n Baby, I need you...\n[outro]\nCome back..."
            }}

        You may decide to add a third verse.
        It is at your discretion.
        However, the song must have at least 2 verses and an outro.
        You do not have to mention anything around baby, love or missing someone. My example was simple to show structure and not the idea of the song lyrics you write.

        REMEMBER .json DOES NOT SUPPORT MULTI LINE STRING SO YOUR LYRICS AND TAGS NEED TO USE THE ESCAPE SEQUENCE \n FOR NEW LINES!
        REMEMBER .json DOES NOT SUPPORT MULTI LINE STRING SO YOUR LYRICS AND TAGS NEED TO USE THE ESCAPE SEQUENCE \n FOR NEW LINES!
        REMEMBER .json DOES NOT SUPPORT MULTI LINE STRING SO YOUR LYRICS AND TAGS NEED TO USE THE ESCAPE SEQUENCE \n FOR NEW LINES!

        To drive the point home that the song maye have varying structure and that it is at your discretion. Verse can have more than 1 lines. Here is another example.
         Another Example Format:
            {{
                "title": "Silent",
                "tags": "female voice, r&b, dreamy, 95 BPM",
                "lyrics": "[intro-medium]\n\n[verse]\nCold...\nRain...\nFalls...\nDown...\nWait...\nFor...\nThe...\nLight...\n\n[chorus]\nStay...\nWith...\nMe...\nJust...\nHold...\nThe...\nSilent...\nSea...\n\n[inst-short]\n\n[verse]\nTime...\nMoves...\nSlow...\nWhere...\nDo...\nWe...\nGo...\nNow...\n\n[chorus]\nStay...\nWith...\nMe...\nJust...\nHold...\nThe...\nSilent...\nSea...\n\n[outro-medium]"
            }}
        
        Here is the user's idea:

        {user_prompt}.

        Give me the output.
        """
        },
    ]

    while attempts < max_retries:
        try:
            # Setting temperature low (0.2) to stay closer to the requested format
            result = pipe(messages, max_new_tokens=600, temperature=0.2, do_sample=True)
            raw_content = result[0]["generated_text"][-1]["content"]

            # MERCENARY REGEX: 
            # \{ is the start. 
            # .*? is NON-GREEDY (stops at the FIRST possible closing brace).
            # re.DOTALL ensures it catches newlines inside the JSON.
            match = re.search(r'\{.*?\}', raw_content, re.DOTALL)
            
            if match:
                clean_json = match.group(0)
                data = json.loads(clean_json)
                
                # Verify the model didn't hallucinate the keys
                if all(k in data for k in ["title", "tags", "lyrics"]):
                    return data # Immediate exit on success
            
        except Exception as e:
            print(f"Attempt {attempts+1} failed to parse JSON. Error: {e}")
        
        attempts += 1

    # If we get here, all retries failed
    return {
        "error": "Failed to generate valid structure after 3 attempts.",
        "raw_fallback": raw_content[:200] if 'raw_content' in locals() else "No output"
    }

# Start RunPod serverless
runpod.serverless.start({"handler": handler})