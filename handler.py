import os
import re
import json
import torch
import runpod
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import pipeline, AutoTokenizer

HF_TOKEN = os.environ.get("HF_TOKEN")
model_id = "HuggingFaceTB/SmolLM3-3B"
pipe = None

def download_models():
    # Using a clean path in /tmp
    cache_dir = os.environ.get("HF_HOME", "/tmp")
    model_dir = os.path.join(cache_dir, "SmolLM3")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    # Use their staggered download strategy
    for filename in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
        hf_hub_download(repo_id=model_id, filename=filename, local_dir=model_dir, token=HF_TOKEN)
    # Download everything to local_dir
    snapshot_download(repo_id=model_id, local_dir=model_dir, token=HF_TOKEN)
    return model_dir

def init_pipeline():
    global pipe
    model_dir = download_models()
    # Load tokenizer from the LOCAL directory to avoid network/auth issues here
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # RunPod requires an integer for device
    device_id = 0 if torch.cuda.is_available() else -1
    
    pipe = pipeline(
        "text-generation", 
        model=model_dir, 
        tokenizer=tokenizer, 
        device=device_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    print("✅ Pipeline Initialized")

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
            result = pipe(messages, max_new_tokens=600, temperature=0.2, do_sample=True)
            raw_content = result[0]["generated_text"][-1]["content"]

            # Non-greedy regex to catch the first JSON block
            match = re.search(r'\{.*?\}', raw_content, re.DOTALL)
            if match:
                output = json.loads(match.group(0))  # Validate JSON
                return {"output": output}
            
        except Exception as e:
            print(f"Attempt {attempts+1} failed: {e}")
        
        attempts += 1

    return {"error": "Failed to generate JSON", "raw": raw_content[:100] if 'raw_content' in locals() else ""}

# Initialize before starting the serverless loop
init_pipeline()
runpod.serverless.start({"handler": handler})