
FROM python:3.13.15-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# Install Node.js and the Claude Code CLI (Claude via the Claude.ai subscription token)
RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm ca-certificates \
    && npm install -g @anthropic-ai/claude-code \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV HF_HOME=/app/huggingface

# Download the model at build time
RUN python -c "\
from langchain_huggingface import HuggingFaceEmbeddings; \
HuggingFaceEmbeddings( \
    model_name='HarishMaths/Hotel-Policy-Embedding', \
    model_kwargs={'device': 'cpu'}, \
    encode_kwargs={'normalize_embeddings': True} \
)"

# Force offline mode at runtime so it uses the cached model
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1

CMD ["gunicorn", "-w", "1", "--timeout", "120", "-b", "0.0.0.0:7860", "app:rag_api"]
