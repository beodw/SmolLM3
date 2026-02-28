import gc
import os
import torch
import runpod
import outlines
from pydantic import BaseModel
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_TOKEN = os.environ.get("HF_TOKEN")
model_id = "HuggingFaceTB/SmolLM3-3B"
generator = None

# Define the schema for deterministic output
class SongSchema(BaseModel):
    title: str
    tags: str
    lyrics: str

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
    global generator
    model_dir = download_models()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Outlines needs the model and tokenizer objects
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, 
        device_map="auto", 
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Create the guided generator
    outlines_model = outlines.models.Transformers(model, tokenizer)
    generator = outlines.generate.json(outlines_model, SongSchema)
    
    print("✅ Outlines Generator Initialized")

def cleanup():
    torch.cuda.empty_cache() 
    torch.cuda.ipc_collect()
    gc.collect()

def handler(job):
    cleanup()
    job_input = job["input"]
    user_prompt = job_input.get("prompt", "A smooth r&b song")
    
    # We simplify the prompt because the schema handles the structure
    prompt = f"<|user|>\nGenerate a song idea: {user_prompt}\nRules: Every line in 'lyrics' must end with '...'. Title max 2 words.<|assistant|>\n"
    
    try:
        # generator() returns a Pydantic object based on SongSchema
        structured_output = generator(prompt, max_tokens=600, temperature=0.1)
        
        cleanup()
        # Convert Pydantic object to dict for RunPod return
        return {"refresh_worker": False, "output": structured_output.model_dump()}
            
    except Exception as e:
        print(f"Generation failed: {e}")
        cleanup()
        return {"refresh_worker": False, "error": str(e)}

init_pipeline()
runpod.serverless.start({"handler": handler})