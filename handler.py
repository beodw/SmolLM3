import gc
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
def cleanup():
    torch.cuda.empty_cache() 
    torch.cuda.ipc_collect()
    gc.collect()

def handler(job):
    cleanup()
    job_input = job["input"]
    user_prompt = job_input.get("prompt", "A smooth r&b song")
    max_retries = 3
    attempts = 0


    messages = [
        {"role": "system", "content": "/no_think"},
        {"role": "user", "content": 
            f"""Generate ONLY a raw JSON object for the song idea: {user_prompt}

                Rules:
                1. Every line in "lyrics" MUST end with '...' (three dots).
                2. Use \\n for new lines.
                3. Include "title", "tags", and "lyrics". The title must be no more than 2 words.
                4. Output ONLY the JSON. No markdown. No backticks.

                Example Structure:
                {{
                    "title": "Short Title",
                    "tags": "female voice, style, mood, BPM",
                    "lyrics": "[intro-short]\\nThe sun is high...\\nThe wind is dry...\\n[verse]\\nWalking through the golden heat...\\nDust beneath my weary feet...\\n[chorus]\\nOh the desert calls my name...\\nNothing ever stays the same...\\n[outro-short]"
                }}

                JSON Output:
                {{
                        }}
            """
        }
    ]
    
    while attempts < max_retries:
        try:
            result = pipe(messages, max_new_tokens=600, temperature=0.1, do_sample=True)
            raw_content = result[0]["generated_text"][-1]["content"]

            # Non-greedy regex to catch the first JSON block
            match = re.search(r"{.*\}", raw_content, re.DOTALL)
            if match:
                output = json.loads(match.group(0))  # Validate JSON
                cleanup()
                return {"refresh_worker": True, "output": output}
            
        except Exception as e:
            print(f"Attempt {attempts+1} failed: {e}")
        
        attempts += 1
        
    cleanup()
    return {"refresh_worker": True,"error": "Failed to generate JSON", "raw": raw_content[:100] if 'raw_content' in locals() else ""}

# Initialize before starting the serverless loop
init_pipeline()
runpod.serverless.start({"handler": handler})