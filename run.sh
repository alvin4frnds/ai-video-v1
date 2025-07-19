#!/bin/bash
# Video Generation Environment Launcher

echo "🎬 AI Video Generation Pipeline"
echo "=============================="

# Activate conda environment
echo "Activating conda environment: video-generation..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate video-generation

# Verify environment
echo "Current environment: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"

# Launch application
echo "Starting Gradio interface..."
echo "📱 Open http://localhost:8003 in your browser"
echo ""

python app.py