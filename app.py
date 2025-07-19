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
        all_batch_analysis = []  # Collect all batch analysis data for UI
        
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
            
            image_result = generator.generate_image(scene)
            
            # Sub-progress for face analysis
            progress(scene_progress_start + 0.8 * (scene_progress_end - scene_progress_start), 
                    f"🔍 Scene {i+1}: Analyzing faces and selecting best image...")
            
            # Handle both old and new return formats
            if isinstance(image_result, dict):
                image_path = image_result['final_path']
                batch_analysis = image_result.get('batch_analysis', [])
                scene_id = image_result.get('scene_id', i+1)
                
                # Add scene info to batch analysis
                for analysis in batch_analysis:
                    analysis['scene_id'] = scene_id
                    analysis['scene_description'] = scene['description']
                    analysis['scene_prompt'] = scene['prompt']
                
                all_batch_analysis.extend(batch_analysis)
            else:
                # Old format compatibility
                image_path = image_result
            
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
        
        return video_path, generated_images, all_batch_analysis, "Video generation completed"
        
    except Exception as e:
        logging.error(f"Error during video generation: {str(e)}")
        progress(1.0, f"Error: {str(e)}")
        return None, [], [], f"Error: {str(e)}"

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

def create_batch_analysis_display(batch_analysis):
    """Create detailed display of all batch images with face analysis scores"""
    if not batch_analysis:
        return "No batch analysis data available.", []
    
    # Group by scene
    scenes = {}
    for analysis in batch_analysis:
        scene_id = analysis.get('scene_id', 0)
        if scene_id not in scenes:
            scenes[scene_id] = []
        scenes[scene_id].append(analysis)
    
    # Create detailed report
    report_lines = []
    report_lines.append("# 🎬 Batch Image Analysis Report")
    report_lines.append(f"**Total Images Generated**: {len(batch_analysis)}")
    report_lines.append(f"**Scenes Processed**: {len(scenes)}")
    report_lines.append("")
    
    # Collect all image paths for gallery
    all_image_paths = []
    
    for scene_id in sorted(scenes.keys()):
        scene_data = scenes[scene_id]
        if not scene_data:
            continue
            
        # Scene header
        scene_desc = scene_data[0].get('scene_description', f'Scene {scene_id}')
        report_lines.append(f"## 🎭 Scene {scene_id}: {scene_desc[:60]}...")
        report_lines.append("")
        
        # Sort by score (highest first)
        scene_data.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        for i, analysis in enumerate(scene_data):
            rank_icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
            selected_icon = " ⭐ **SELECTED**" if analysis.get('is_selected', False) else ""
            
            filename = analysis.get('filename', 'Unknown')
            score = analysis.get('score', 0)
            face_count = analysis.get('face_count', 0)
            quality_score = analysis.get('quality_score', 0)
            has_realistic = analysis.get('has_realistic_face', False)
            methods = analysis.get('detection_methods', [])
            error = analysis.get('error', None)
            clothing_consistency = analysis.get('clothing_consistency', 0.5)
            clothing_colors = analysis.get('clothing_colors', [])
            clothing_error = analysis.get('clothing_error', None)
            
            report_lines.append(f"### {rank_icon} {filename}{selected_icon}")
            report_lines.append(f"- **Overall Score**: {score:.3f}")
            report_lines.append(f"- **Face Count**: {face_count}")
            report_lines.append(f"- **Face Quality**: {quality_score:.3f}")
            report_lines.append(f"- **Realistic Face**: {'✅ Yes' if has_realistic else '❌ No'}")
            report_lines.append(f"- **Detection Methods**: {', '.join(methods) if methods else 'None'}")
            report_lines.append(f"- **Clothing Consistency**: {clothing_consistency:.3f} {'✅' if clothing_consistency > 0.7 else '⚠️' if clothing_consistency > 0.4 else '❌'}")
            if clothing_colors:
                color_str = ', '.join([f"RGB({r},{g},{b})" for r, g, b in clothing_colors[:3]])
                report_lines.append(f"- **Dominant Colors**: {color_str}")
            
            if error:
                report_lines.append(f"- **Face Analysis Error**: ⚠️ {error}")
            if clothing_error:
                report_lines.append(f"- **Clothing Analysis Error**: ⚠️ {clothing_error}")
            
            # Individual face details
            faces = analysis.get('faces', [])
            if faces:
                report_lines.append(f"- **Face Details**:")
                for j, face in enumerate(faces):
                    conf = face.get('confidence', 0)
                    method = face.get('method', 'Unknown')
                    area = face.get('area', 0)
                    report_lines.append(f"  - Face {j+1}: {method}, confidence: {conf:.2f}, area: {area}px")
            
            report_lines.append("")
            
            # Add to gallery
            img_path = analysis.get('path', '')
            if img_path and os.path.exists(img_path):
                all_image_paths.append(img_path)
        
        report_lines.append("---")
        report_lines.append("")
    
    return "\n".join(report_lines), all_image_paths

def create_batch_summary_stats(batch_analysis):
    """Create summary statistics from batch analysis"""
    if not batch_analysis:
        return "No data"
    
    total_images = len(batch_analysis)
    total_faces = sum(analysis.get('face_count', 0) for analysis in batch_analysis)
    avg_quality = sum(analysis.get('quality_score', 0) for analysis in batch_analysis) / total_images if total_images > 0 else 0
    realistic_count = sum(1 for analysis in batch_analysis if analysis.get('has_realistic_face', False))
    avg_clothing_consistency = sum(analysis.get('clothing_consistency', 0.5) for analysis in batch_analysis) / total_images if total_images > 0 else 0.5
    high_consistency_count = sum(1 for analysis in batch_analysis if analysis.get('clothing_consistency', 0.5) > 0.7)
    
    methods_used = set()
    for analysis in batch_analysis:
        methods_used.update(analysis.get('detection_methods', []))
    
    stats = f"""📊 **Batch Analysis Summary**
    
- **Total Images**: {total_images}
- **Total Faces Detected**: {total_faces}
- **Average Face Quality**: {avg_quality:.3f}
- **Realistic Faces**: {realistic_count}/{total_images} ({realistic_count/total_images*100:.1f}%)
- **Average Clothing Consistency**: {avg_clothing_consistency:.3f}
- **High Consistency Images**: {high_consistency_count}/{total_images} ({high_consistency_count/total_images*100:.1f}%)
- **Detection Methods Used**: {', '.join(sorted(methods_used)) if methods_used else 'None'}
"""
    return stats

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
            label="Final Selected Frames (6 frames)",
            show_label=True,
            elem_id="gallery",
            columns=3,
            rows=2,
            object_fit="contain",
            height="auto"
        )
    
    with gr.Row():
        with gr.Column(scale=1):
            batch_stats = gr.Markdown(
                label="Batch Analysis Summary",
                value="Generate a video to see face detection statistics..."
            )
        with gr.Column(scale=2):
            batch_report = gr.Markdown(
                label="Detailed Face Analysis Report",
                value="Generate a video to see detailed face analysis for all generated images..."
            )
    
    with gr.Row():
        batch_gallery = gr.Gallery(
            label="All Generated Images with Face & Clothing Analysis (36 total: 6 per scene)",
            show_label=True,
            elem_id="batch_gallery",
            columns=6,
            rows=6,
            object_fit="contain",
            height="auto"
        )
    
    # Remove timer since we no longer have logs display
    
    # Generate button click handler
    def handle_generation(prompt):
        video_path, generated_images, batch_analysis, logs = generate_video_pipeline(prompt)
        gallery_data = create_still_preview(generated_images)
        batch_report, batch_images = create_batch_analysis_display(batch_analysis)
        batch_stats = create_batch_summary_stats(batch_analysis)
        return video_path, gallery_data, batch_report, batch_images, batch_stats
    
    # Generate button click handler
    generate_btn.click(
        fn=handle_generation,
        inputs=[prompt_input],
        outputs=[video_output, still_gallery, batch_report, batch_gallery, batch_stats]
    )
    
    # Random test button click handler
    def handle_random_test():
        random_prompt = generate_random_test_prompt()
        video_path, generated_images, batch_analysis, logs = generate_video_pipeline(random_prompt)
        gallery_data = create_still_preview(generated_images)
        batch_report, batch_images = create_batch_analysis_display(batch_analysis)
        batch_stats = create_batch_summary_stats(batch_analysis)
        return random_prompt, video_path, gallery_data, batch_report, batch_images, batch_stats
    
    random_btn.click(
        fn=handle_random_test,
        outputs=[prompt_input, video_output, still_gallery, batch_report, batch_gallery, batch_stats]
    )
    
    # Clean button click handler
    def handle_clean():
        message, video, gallery = clean_output_folder()
        return message, video, gallery, "Generate a video to see detailed face analysis...", [], "Generate a video to see face detection statistics..."
    
    clean_btn.click(
        fn=handle_clean,
        outputs=[progress_display, video_output, still_gallery, batch_report, batch_gallery, batch_stats]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=8003,
        share=False,
        show_error=True
    )