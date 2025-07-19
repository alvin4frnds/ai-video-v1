import gradio as gr
import time
import threading
from datetime import datetime
import logging
import sys
from io import StringIO
import queue
import os

from video_generator import VideoGenerator

class LogCapture:
    def __init__(self):
        self.log_queue = queue.Queue()
        self.setup_logging()
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        
        handler = QueueHandler(self.log_queue)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', '%H:%M:%S')
        handler.setFormatter(formatter)
        
        logger = logging.getLogger()
        logger.addHandler(handler)
    
    def get_logs(self):
        logs = []
        while not self.log_queue.empty():
            try:
                logs.append(self.log_queue.get_nowait())
            except queue.Empty:
                break
        return logs

class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue
    
    def emit(self, record):
        self.log_queue.put(self.format(record))

log_capture = LogCapture()

def generate_video_pipeline(prompt, progress=gr.Progress()):
    """Main video generation pipeline with progress tracking"""
    
    generator = VideoGenerator()
    
    # Initialize progress
    progress(0, "Starting video generation pipeline...")
    logging.info(f"Starting video generation for prompt: '{prompt}'")
    
    try:
        # Step 1: Analyze prompt (10%)
        progress(0.1, "Analyzing text prompt...")
        logging.info("Step 1/4: Analyzing text prompt")
        scenes = generator.analyze_prompt(prompt)
        logging.info(f"Identified {len(scenes)} scenes")
        
        # Step 2: Plan sequences (20%)
        progress(0.2, "Planning scene sequences...")
        logging.info("Step 2/4: Planning scene sequences")
        scene_plan = generator.plan_sequences(scenes)
        logging.info(f"Created sequence plan with {len(scene_plan)} frames")
        
        # Step 3: Generate images (20-80%)
        progress(0.2, "Generating still frame images...")
        logging.info("Step 3/4: Generating still frame images")
        generated_images = []
        
        for i, scene in enumerate(scene_plan):
            frame_progress = 0.2 + (0.6 * (i + 1) / len(scene_plan))
            progress(frame_progress, f"Generating image {i+1}/{len(scene_plan)}: {scene['description'][:50]}...")
            logging.info(f"Generating frame {i+1}/{len(scene_plan)}: {scene['description']}")
            
            image_path = generator.generate_image(scene)
            generated_images.append({
                'path': image_path,
                'prompt': scene['prompt'],
                'description': scene['description']
            })
            logging.info(f"Generated image saved to: {image_path}")
        
        # Step 4: Create transitions (80-100%)
        progress(0.8, "Creating video transitions...")
        logging.info("Step 4/4: Creating video transitions")
        video_path = generator.create_video_with_transitions(generated_images)
        progress(1.0, "Video generation complete!")
        logging.info(f"Video generation complete! Saved to: {video_path}")
        
        return video_path, generated_images, get_current_logs()
        
    except Exception as e:
        logging.error(f"Error during video generation: {str(e)}")
        progress(1.0, f"Error: {str(e)}")
        return None, [], get_current_logs()

def get_current_logs():
    """Get current log entries for display"""
    logs = log_capture.get_logs()
    return "\n".join(logs) if logs else "No logs yet..."

def update_logs():
    """Update log display in real-time"""
    return get_current_logs()

def create_still_preview(images):
    """Create preview gallery of generated stills"""
    if not images:
        return []
    
    preview_data = []
    for i, img_data in enumerate(images):
        if os.path.exists(img_data['path']):
            preview_data.append((img_data['path'], f"Frame {i+1}: {img_data['description'][:100]}..."))
    
    return preview_data

# Create Gradio interface
with gr.Blocks(title="AI Video Generation Pipeline", theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# 🎬 AI Video Generation Pipeline")
    gr.Markdown("Convert text prompts into videos through scene-based generation")
    
    with gr.Row():
        with gr.Column(scale=2):
            prompt_input = gr.Textbox(
                label="Text Prompt",
                placeholder="Enter your story or scene description...",
                lines=4
            )
            generate_btn = gr.Button("🚀 Generate Video", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            progress_display = gr.Textbox(
                label="Current Status",
                value="Ready to generate...",
                interactive=False
            )
    
    with gr.Row():
        with gr.Column():
            video_output = gr.Video(label="Generated Video")
        
        with gr.Column():
            logs_output = gr.Textbox(
                label="🖥️ System Logs",
                lines=15,
                value="[SYSTEM] Ready for video generation...",
                interactive=False,
                show_copy_button=True
            )
    
    with gr.Row():
        still_gallery = gr.Gallery(
            label="Generated Still Frames",
            show_label=True,
            elem_id="gallery",
            columns=3,
            rows=2,
            object_fit="contain",
            height="auto"
        )
    
    # Auto-refresh logs every 2 seconds
    timer = gr.Timer(2)
    timer.tick(
        fn=update_logs,
        outputs=logs_output
    )
    
    # Generate button click handler
    generate_btn.click(
        fn=generate_video_pipeline,
        inputs=[prompt_input],
        outputs=[video_output, still_gallery, logs_output]
    ).then(
        fn=create_still_preview,
        inputs=[still_gallery],
        outputs=[still_gallery]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=8003,
        share=False,
        show_error=True
    )