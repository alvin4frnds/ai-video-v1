import gradio as gr
import time
import threading
from datetime import datetime
import logging
import sys
import os
import random
import shutil

from video_generator import VideoGenerator

# Simple logging setup for console output
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)

def generate_random_test_prompt():
    """Generate a random test prompt for empty input"""
    colors = ["red", "blue", "green", "yellow", "purple", "pink", "black", "white", "orange", "brown", "gray"]
    locations = ["sidewalk", "street", "park", "mall", "market"]
    
    # Use YmdH seed for consistent randomization within the hour
    seed = int(datetime.now().strftime("%Y%m%d%H"))
    random.seed(seed)
    
    color = random.choice(colors)
    location = random.choice(locations)
    
    prompt = f"woman wearing {color} walking {location}"
    logging.info(f"Generated random test prompt: {prompt} (seed: {seed})")
    return prompt

def generate_video_pipeline(prompt, progress=gr.Progress()):
    """Main video generation pipeline with progress tracking"""
    
    # Check if prompt is empty and generate random test prompt
    if not prompt or prompt.strip() == "":
        prompt = generate_random_test_prompt()
        logging.info("Empty prompt detected, using random test prompt")
    
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
            scene_progress_start = 0.2 + (0.6 * i / len(scene_plan))
            scene_progress_end = 0.2 + (0.6 * (i + 1) / len(scene_plan))
            
            # Update progress for scene start
            progress(scene_progress_start, f"🎬 Scene {i+1}/{len(scene_plan)}: Starting batch generation...")
            logging.info(f"=" * 60)
            logging.info(f"🎬 SCENE {i+1}/{len(scene_plan)} - BATCH IMAGE GENERATION")
            logging.info(f"📝 Description: {scene['description']}")
            logging.info(f"🎯 Enhanced prompt: {scene['prompt'][:150]}{'...' if len(scene['prompt']) > 150 else ''}")
            logging.info(f"=" * 60)
            
            # Sub-progress updates for batch generation
            progress(scene_progress_start + 0.1 * (scene_progress_end - scene_progress_start), 
                    f"📸 Scene {i+1}: Generating 6 candidate images...")
            
            image_path = generator.generate_image(scene)
            
            # Sub-progress for face analysis
            progress(scene_progress_start + 0.8 * (scene_progress_end - scene_progress_start), 
                    f"🔍 Scene {i+1}: Analyzing faces and selecting best image...")
            
            generated_images.append({
                'path': image_path,
                'prompt': scene['prompt'],
                'description': scene['description'],
                'duration': scene.get('duration', 3.0),
                'transition_type': scene.get('transition_type', 'fade')
            })
            
            # Complete scene progress
            progress(scene_progress_end, f"✅ Scene {i+1} complete: {os.path.basename(image_path) if image_path else 'No image generated'}")
            logging.info(f"✅ SCENE {i+1} COMPLETED - Final image: {image_path}")
            logging.info("")  # Add spacing between scenes
        
        # Step 4: Create transitions (80-100%)
        progress(0.8, "Creating video transitions...")
        logging.info("Step 4/4: Creating video transitions")
        video_path = generator.create_video_with_transitions(generated_images)
        progress(1.0, "Video generation complete!")
        logging.info(f"Video generation complete! Saved to: {video_path}")
        
        return video_path, generated_images, "Video generation completed"
        
    except Exception as e:
        logging.error(f"Error during video generation: {str(e)}")
        progress(1.0, f"Error: {str(e)}")
        return None, [], f"Error: {str(e)}"

# Removed log display functions since logs UI component was removed

def create_still_preview(images):
    """Create preview gallery of generated stills"""
    if not images:
        return []
    
    # Handle case where images might be a list of dictionaries
    if isinstance(images, list) and len(images) > 0:
        if isinstance(images[0], dict):
            # Extract image paths from dictionaries
            preview_data = []
            for i, img_data in enumerate(images):
                if 'path' in img_data and os.path.exists(img_data['path']):
                    preview_data.append(img_data['path'])
            return preview_data
        else:
            # Already in correct format
            return images
    
    return []

def clean_output_folder():
    """Clean the output folder and return status message"""
    try:
        output_dir = "output"
        
        # Count files before cleaning
        total_files = 0
        total_size = 0
        video_count = 0
        frame_count = 0
        batch_dirs = 0
        
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file != '.gitkeep':  # Don't count .gitkeep
                    file_path = os.path.join(root, file)
                    total_files += 1
                    total_size += os.path.getsize(file_path)
                    
                    # Count by type
                    if file.endswith(('.mp4', '.avi', '.mov')):
                        video_count += 1
                    elif file.endswith(('.png', '.jpg', '.jpeg')):
                        frame_count += 1
            
            # Count batch directories
            for dir_name in dirs:
                if dir_name.startswith('batch_scene_'):
                    batch_dirs += 1
        
        if total_files == 0:
            logging.info("🧹 Output folder is already clean")
            return "✅ Output folder is already clean - no files to remove", None, []
        
        logging.info(f"🧹 Starting cleanup: {total_files} files, {video_count} videos, {frame_count} images, {batch_dirs} batch dirs")
        
        # Clean the folders
        frames_dir = os.path.join(output_dir, "frames")
        videos_dir = os.path.join(output_dir, "videos")
        
        # Remove all contents except .gitkeep
        removed_files = 0
        removed_dirs = 0
        
        for root, dirs, files in os.walk(output_dir, topdown=False):
            for file in files:
                if file != '.gitkeep':
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                    removed_files += 1
            
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                if dir_name not in ['frames', 'videos'] and os.path.exists(dir_path):
                    try:
                        # Check if directory is empty (except for subdirs we want to keep)
                        if not any(os.listdir(dir_path)) or dir_name.startswith('batch_scene_'):
                            shutil.rmtree(dir_path)
                            removed_dirs += 1
                    except OSError:
                        pass  # Directory might not be empty or have permission issues
        
        # Convert size to human readable
        if total_size >= 1024 * 1024:
            size_str = f"{total_size / (1024 * 1024):.1f} MB"
        elif total_size >= 1024:
            size_str = f"{total_size / 1024:.1f} KB"
        else:
            size_str = f"{total_size} bytes"
        
        message = f"🧹 Cleanup complete: {removed_files} files removed ({size_str} freed)"
        if removed_dirs > 0:
            message += f", {removed_dirs} directories removed"
        
        logging.info(message)
        
        return message, None, []
        
    except Exception as e:
        error_msg = f"❌ Error cleaning output folder: {str(e)}"
        logging.error(error_msg)
        return error_msg, None, []

# Create Gradio interface with custom CSS for sans-serif fonts
css = """
* {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif !important;
}

.gr-button {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif !important;
}

.gr-textbox textarea, .gr-textbox input {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif !important;
}

.gr-markdown {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif !important;
}

.gr-label {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif !important;
}
"""

with gr.Blocks(title="AI Video Generation Pipeline", theme=gr.themes.Monochrome(), css=css) as demo:
    gr.Markdown("# 🎬 AI Video Generation Pipeline")
    gr.Markdown("Convert text prompts into videos through scene-based generation")
    
    with gr.Row():
        with gr.Column(scale=2):
            prompt_input = gr.Textbox(
                label="Text Prompt",
                placeholder="Enter your story or scene description... (leave empty for random test)",
                lines=4
            )
            with gr.Row():
                generate_btn = gr.Button("🚀 Generate Video", variant="primary", size="lg")
                random_btn = gr.Button("🎲 Random Test", variant="secondary", size="sm")
                clean_btn = gr.Button("🧹 Clean Output", variant="secondary", size="sm")
        
        with gr.Column(scale=1):
            progress_display = gr.Textbox(
                label="Current Status",
                value="Ready to generate...",
                interactive=False
            )
    
    with gr.Row():
        video_output = gr.Video(label="Generated Video")
    
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
    
    # Remove timer since we no longer have logs display
    
    # Generate button click handler
    def handle_generation(prompt):
        video_path, generated_images, logs = generate_video_pipeline(prompt)
        gallery_data = create_still_preview(generated_images)
        return video_path, gallery_data
    
    # Generate button click handler
    generate_btn.click(
        fn=handle_generation,
        inputs=[prompt_input],
        outputs=[video_output, still_gallery]
    )
    
    # Random test button click handler
    def handle_random_test():
        random_prompt = generate_random_test_prompt()
        video_path, generated_images, logs = generate_video_pipeline(random_prompt)
        gallery_data = create_still_preview(generated_images)
        return random_prompt, video_path, gallery_data
    
    random_btn.click(
        fn=handle_random_test,
        outputs=[prompt_input, video_output, still_gallery]
    )
    
    # Clean button click handler
    clean_btn.click(
        fn=clean_output_folder,
        outputs=[progress_display, video_output, still_gallery]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=8003,
        share=False,
        show_error=True
    )