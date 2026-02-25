import os
import re
import json
import torch
import runpod
from huggingface_hub import snapshot_download
from transformers import pipeline, AutoTokenizer

HF_TOKEN = os.environ.get("HF_TOKEN")
model_id = "HuggingFaceTB/SmolLM3-3B"
pipe = None

def download_models():
    # Using a clean path in /tmp
    model_dir = os.path.join("/tmp", "SmolLM3")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

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
        low_cpu_mem_usage=True
    )
    print("✅ Pipeline Initialized")

def handler(job):
    return {"output":"success"}
    job_input = job["input"]
    user_prompt = job_input.get("prompt", "A smooth r&b song")
    max_retries = 3
    attempts = 0

    messages = [
        {"role": "system", "content": "You are a rigid API response generator. Output ONLY a raw JSON object. No markdown."},
        {"role": "user", "content": f"""
        Task: Transform idea into JSON.
        Rules: 
        1. title: 1-2 words.
        2. tags: single line.
        3. lyrics: use [section], lines end with '...', must rhyme. Use \\n for new lines.
        
        Idea: {user_prompt}
        
        Format:
        {{
            "title": "Song Title",
            "tags": "tags here",
            "lyrics": "[intro]\\nLine...\\n"
        }}
        """}
    ]
    
    while attempts < max_retries:
        try:
            result = pipe(messages, max_new_tokens=600, temperature=0.2, do_sample=True)
            raw_content = result[0]["generated_text"][-1]["content"]

            # Non-greedy regex to catch the first JSON block
            match = re.search(r'\{.*?\}', raw_content, re.DOTALL)
            if match:
                return {"output": match.group(0)}
            
        except Exception as e:
            print(f"Attempt {attempts+1} failed: {e}")
        
        attempts += 1

    return {"error": "Failed to generate JSON", "raw": raw_content[:100] if 'raw_content' in locals() else ""}

# Initialize before starting the serverless loop
init_pipeline()
runpod.serverless.start({"handler": handler})