import gc
import os
import torch
import runpod
import outlines
from pydantic import BaseModel, field_validator
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_TOKEN = os.environ.get("HF_TOKEN")
model_id = "HuggingFaceTB/SmolLM3-3B"
model = None

class SongSchema(BaseModel):
    title: str
    genre: str
    tags: str
    lyrics: str

    @field_validator('title')
    @classmethod
    def enforce_title_limit(cls, v):
        # Street Smart: If the model gives 3+ words, just chop it.
        words = v.split()
        return " ".join(words[:2]) if len(words) > 2 else v

    @field_validator('lyrics')
    @classmethod
    def enforce_dots(cls, v):
        # The "Dumb Simple" fix: Ensure every non-empty line ends with '...'
        lines = [line.strip() for line in v.split('\n') if line.strip()]
        fixed_lines = [line if line.endswith('...') else f"{line}..." for line in lines]
        return "\n".join(fixed_lines)

    @field_validator('tags')
    @classmethod
    def ensure_genre_in_tags(cls, v, info):
        # Guarantee the genre is always the first tag
        genre = info.data.get('genre', 'Unknown')
        if genre.lower() not in v.lower():
            return f"{genre}, {v}"
        return v

def download_models():
    cache_dir = os.environ.get("HF_HOME", "/tmp")
    model_dir = os.path.join(cache_dir, "SmolLM3")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    for filename in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
        hf_hub_download(repo_id=model_id, filename=filename, local_dir=model_dir, token=HF_TOKEN)
    snapshot_download(repo_id=model_id, local_dir=model_dir, token=HF_TOKEN)
    return model_dir

def init_pipeline():
    global model
    model_dir = download_models()
    raw_model = AutoModelForCausalLM.from_pretrained(
        model_dir, device_map="auto", 
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = outlines.from_transformers(raw_model, tokenizer)
    print("✅ Outlines Model Initialized")

def cleanup():
    torch.cuda.empty_cache() 
    torch.cuda.ipc_collect()
    gc.collect()

def handler(job):
    cleanup()
    job_input = job["input"]
    user_prompt = job_input.get("prompt", "A smooth r&b song")
    
    # We simplified the instructions because the Python validators do the heavy lifting now
    prompt = f"<|user|>\nGenerate a song: {user_prompt}\nRules: Title max 2 words. Every line ends with '...'.<|assistant|>\n"
    
    try:
        output_data = model(prompt, output_type=SongSchema, max_new_tokens=600, temperature=0.1, do_sample=True, repetition_penalty=1.1)
        
        # Pydantic 2026 validation logic
        if isinstance(output_data, str):
            structured_output = SongSchema.model_validate_json(output_data)
        else:
            structured_output = SongSchema.model_validate(output_data)

        cleanup()
        return {"refresh_worker": False, "output": structured_output.model_dump()}
            
    except Exception as e:
        print(f"Generation failed: {e}")
        cleanup()
        return {"refresh_worker": False, "error": str(e)}

init_pipeline()
runpod.serverless.start({"handler": handler})