# Mixtral Setup Instructions

## Prerequisites

1. **Install Ollama**
   ```bash
   # On Linux/WSL
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Or download from https://ollama.ai/download
   ```

2. **Start Ollama Service**
   ```bash
   ollama serve
   ```

3. **Pull Mixtral Model**
   ```bash
   # Pull the Mixtral model (this will take some time - ~26GB)
   ollama pull mixtral
   
   # Alternative: Use smaller model for testing
   ollama pull mixtral:7b
   ```

## Verification

Test that Mixtral is working:
```bash
# Test the model
ollama run mixtral "Hello, how are you?"
```

## Configuration

The video generator will automatically:
- Connect to Ollama at `http://localhost:11434`
- Use Mixtral for scene analysis and prompt enhancement
- Fall back to basic methods if Mixtral is unavailable

## Features Added

✅ **Intelligent Scene Analysis** - Mixtral analyzes narrative structure and identifies optimal scene breaks
✅ **Enhanced Image Prompts** - Converts basic descriptions into detailed, cinematic prompts
✅ **Fallback Support** - Works without Mixtral using basic algorithms
✅ **Real-time Logs** - Shows Mixtral processing status in the UI

## Usage

1. Start Ollama: `ollama serve`
2. Run the app: `./run.sh`
3. The system will automatically use Mixtral for better prompt generation

## Troubleshooting

- **Connection Error**: Ensure Ollama is running on port 11434
- **Model Not Found**: Run `ollama pull mixtral` to download the model
- **Slow Performance**: Mixtral requires significant GPU/CPU resources
- **Fallback Mode**: The system works without Mixtral using basic prompt enhancement