FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl https://ollama.com/install.sh | sh

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pull Phi model (lightweight)
RUN ollama pull phi

# Expose Streamlit port
EXPOSE 8501

# Start Ollama + Streamlit
CMD ollama serve & streamlit run app.py --server.port=8501 --server.address=0.0.0.0
