# Use RunPod's optimized PyTorch base
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# 1. Install system dependencies
# git is needed if any requirements are git-based; build-essential for speed optimizations
RUN apt-get update && apt-get install -y git build-essential && rm -rf /var/lib/apt/lists/*

# 2. Set working directory
WORKDIR /

# 3. Install requirements
# We install these first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -U -r requirements.txt

# 4. Copy your SmolLM3 handler
# This contains your handler(job) and the non-greedy regex logic
COPY handler.py .

# 5. Run the handler
# -u ensures logs appear in the RunPod dashboard immediately
CMD [ "python", "-u", "/handler.py" ]