# Stable Diffusion Setup Instructions

## Prerequisites

1. **Automatic1111 Stable Diffusion WebUI**
   - Download from: https://github.com/AUTOMATIC1111/stable-diffusion-webui
   - Install following their setup guide

2. **Start WebUI with API**
   ```bash
   # Start with API enabled on port 8001
   ./webui.sh --api --port 8001
   
   # Or on Windows
   webui-user.bat --api --port 8001
   
   # For network access (if running on different machine)
   ./webui.sh --api --port 8001 --listen
   ```

## Configuration

The video generator automatically:
- Connects to SD WebUI at `http://192.168.0.199:8001` (configured for your network)
- Auto-detects SD WebUI on common ports and IPs if needed
- Uses optimized settings for video frame generation
- Falls back to placeholder images if SD unavailable

## Default Generation Settings

✅ **Resolution**: 512x768 (portrait format)
✅ **Steps**: 50 (high quality)
✅ **CFG Scale**: 7 (balanced creativity/coherence)
✅ **Sampler**: DPM++ 2M SDE
✅ **Scheduler**: Karras
✅ **Model**: cyberrealistic_v80.safetensors (automatically selected)
✅ **Negative Prompt**: Quality improvements automatically added
✅ **Face Restoration**: Enabled for character scenes

## Features Added

✅ **Real Image Generation** - Uses your SD models instead of placeholders
✅ **Mixtral Enhanced Prompts** - Detailed cinematic prompts for better results
✅ **Automatic Fallback** - Works without SD using placeholder images
✅ **Progress Tracking** - Shows SD generation status in logs
✅ **Model Detection** - Automatically detects current SD model

## Usage

1. Start SD WebUI: `./webui.sh --api --port 8001`
2. Run the app: `./run.sh`
3. The system will automatically use SD for real image generation

## Troubleshooting

- **Connection Error**: Ensure SD WebUI is running with `--api` flag on port 8001
- **Generation Fails**: Check SD WebUI console for errors
- **Slow Performance**: Reduce steps in `sd_client.py` or use faster samplers
- **Memory Issues**: Enable `--lowvram` or `--medvram` flags in SD WebUI
- **Fallback Mode**: System works without SD using placeholder images

## Model Recommendations

For best cinematic results:
- **Realistic Vision** - Great for photorealistic scenes
- **DreamShaper** - Good for fantasy/artistic scenes  
- **Deliberate** - Excellent for detailed characters
- **Anything V5** - Good for anime/stylized content