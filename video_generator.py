import os
import re
import logging
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
from datetime import datetime
from mixtral_client import MixtralClient
from sd_client import StableDiffusionClient

class VideoGenerator:
    def __init__(self):
        self.output_dir = "output"
        self.frames_dir = os.path.join(self.output_dir, "frames")
        self.videos_dir = os.path.join(self.output_dir, "videos")
        
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.videos_dir, exist_ok=True)
        
        # Generate YmdH format seed for consistent results within the hour
        self.base_seed = int(datetime.now().strftime("%Y%m%d%H"))
        logging.info(f"Using base seed: {self.base_seed} (YmdH format for consistency)")
        
        # Initialize Mixtral client
        self.mixtral = MixtralClient()
        self.mixtral.base_seed = self.base_seed  # Share seed for consistency
        
        # Initialize Stable Diffusion client with network IP
        self.sd_client = StableDiffusionClient(base_url="http://192.168.0.199:8001")
        
        logging.info("VideoGenerator initialized")
        logging.info(f"Output directory: {self.output_dir}")
        
        # Check Mixtral availability
        if self.mixtral.check_connection():
            logging.info("Mixtral LLM connected and ready for prompt enhancement")
        else:
            logging.warning("Mixtral LLM unavailable - will use fallback prompt generation")
        
        # Check Stable Diffusion availability
        if self.sd_client.check_connection():
            current_model = self.sd_client.get_current_model()
            logging.info(f"Stable Diffusion WebUI connected - Model: {current_model}")
            
            # Try to set cyberrealistic model as default
            target_model = "cyberrealistic_v80.safetensors [90389105e4]"
            if target_model not in str(current_model):
                logging.info(f"Attempting to switch to preferred model: {target_model}")
                if self.sd_client.set_model(target_model):
                    logging.info("Successfully switched to cyberrealistic model")
                else:
                    logging.warning("Could not switch to cyberrealistic model, using current model")
            
            # Test SD generation to make sure it's working
            logging.info("Testing SD generation with simple prompt...")
            test_path = os.path.join(self.frames_dir, "test_generation.png")
            test_result = self.sd_client.generate_image(
                prompt="test image, simple portrait",
                output_path=test_path,
                width=512,
                height=768,
                steps=5,  # Fast test
                cfg_scale=7,
                seed=self.base_seed
            )
            
            if test_result:
                logging.info("✅ SD test generation successful")
                os.remove(test_path)  # Clean up test file
                self.use_sd = True
            else:
                logging.warning("❌ SD test generation failed - using placeholder mode")
                self.use_sd = False
        else:
            # Try to find SD WebUI on other ports
            found_url = self.sd_client.find_sd_webui()
            if found_url:
                self.sd_client.base_url = found_url
                logging.info(f"✅ Auto-detected SD WebUI at {found_url}")
                self.use_sd = True
            else:
                logging.warning("❌ Stable Diffusion WebUI unavailable - will use placeholder images")
                logging.warning("💡 To use real image generation:")
                logging.warning("   1. Start SD WebUI: ./webui.sh --api --port 8001")
                logging.warning("   2. Or check if it's running on a different port")
                self.use_sd = False
    
    def analyze_prompt(self, prompt):
        """Analyze text prompt and extract scenes/narrative elements using Mixtral"""
        logging.info("Analyzing prompt for narrative structure with Mixtral LLM...")
        
        # Use Mixtral for intelligent scene analysis
        scenes = self.mixtral.analyze_narrative_structure(prompt)
        
        logging.info(f"Extracted {len(scenes)} scenes from prompt")
        
        # Handle both old string format and new dict format for backward compatibility
        if scenes and isinstance(scenes[0], dict):
            # New format with timing data
            for i, scene in enumerate(scenes):
                logging.info(f"Scene {i+1}: {scene['description'][:80]}... (duration: {scene['duration']}s)")
        else:
            # Old format - convert to new format
            scenes_with_timing = []
            for i, scene in enumerate(scenes):
                scenes_with_timing.append({
                    'description': scene,
                    'duration': 3.0,
                    'transition': 'fade' if i > 0 else 'none'
                })
                logging.info(f"Scene {i+1}: {scene[:100]}... (default timing)")
            scenes = scenes_with_timing
        
        # Ensure we have enough scenes for a proper video
        if len(scenes) < 3:
            logging.warning(f"Only {len(scenes)} scenes detected - this may result in a very short video")
        elif len(scenes) > 8:
            logging.info(f"Generated {len(scenes)} scenes - video will be substantial")
        
        return scenes
    
    def plan_sequences(self, scenes):
        """Plan the sequence of still frames needed using Mixtral-enhanced prompts"""
        logging.info("Planning frame sequences with Mixtral-enhanced prompts...")
        
        # Convert scene data for prompt enhancement
        scene_descriptions = []
        for scene in scenes:
            if isinstance(scene, dict):
                scene_descriptions.append(scene['description'])
            else:
                scene_descriptions.append(scene)
        
        # Use Mixtral to generate enhanced prompts for each scene
        enhanced_scenes = self.mixtral.generate_scene_prompts(scene_descriptions)
        
        scene_plan = []
        for i, enhanced_scene in enumerate(enhanced_scenes):
            # Get timing data from original scene if available
            original_scene = scenes[i] if i < len(scenes) else scenes[0]
            duration = original_scene.get('duration', 3.0) if isinstance(original_scene, dict) else 3.0
            transition = original_scene.get('transition', 'fade') if isinstance(original_scene, dict) else 'fade'
            
            frame_data = {
                'scene_id': enhanced_scene['scene_id'],
                'description': enhanced_scene['original_description'],
                'prompt': enhanced_scene['enhanced_prompt'],
                'duration': duration,
                'transition_type': transition
            }
            scene_plan.append(frame_data)
            
            logging.info(f"Planned frame {enhanced_scene['scene_id']}: {frame_data['prompt'][:60]}... (duration: {duration}s)")
        
        return scene_plan
    
    def _create_image_prompt(self, scene_description):
        """Convert scene description to optimized image generation prompt"""
        # Add visual style and quality descriptors
        base_prompt = scene_description
        
        # Add style modifiers for better image generation
        style_additions = [
            "high quality, detailed, cinematic lighting",
            "professional photography, 4k resolution",
            "dramatic composition, vivid colors"
        ]
        
        enhanced_prompt = f"{base_prompt}, {', '.join(style_additions)}"
        return enhanced_prompt
    
    def generate_image(self, scene_data):
        """Generate image for a scene using Stable Diffusion or placeholder"""
        logging.info(f"Generating image for scene {scene_data['scene_id']}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"frame_{scene_data['scene_id']:03d}_{timestamp}.png"
        filepath = os.path.join(self.frames_dir, filename)
        
        if self.use_sd:
            # Use Stable Diffusion for real image generation
            logging.info("Using Stable Diffusion for image generation")
            
            # Generate with SD using optimized settings and consistent seed
            scene_seed = self.base_seed + scene_data['scene_id']  # Different seed per scene but consistent
            logging.info(f"Using seed {scene_seed} for scene {scene_data['scene_id']} (base: {self.base_seed})")
            
            # Try generation with retry on failure
            max_retries = 2
            for attempt in range(max_retries):
                if attempt > 0:
                    logging.info(f"Retrying SD generation (attempt {attempt + 1}/{max_retries})")
                    scene_seed += 1000  # Use different seed for retry
                
                result_path = self.sd_client.generate_image(
                    prompt=scene_data['prompt'],
                    output_path=filepath,
                    width=512,
                    height=768,
                    steps=28,
                    cfg_scale=7,
                    sampler="DPM++ 2M SDE",
                    scheduler="Karras",
                    seed=scene_seed
                )
                
                if result_path:
                    logging.info(f"✅ SD image generated successfully: {filepath} (attempt {attempt + 1})")
                    return result_path
                else:
                    logging.warning(f"❌ SD generation attempt {attempt + 1} failed")
                    if attempt < max_retries - 1:
                        logging.info("Will retry with different seed...")
            
            logging.warning(f"❌ All SD generation attempts failed for scene {scene_data['scene_id']}, falling back to placeholder")
            logging.warning(f"Failed prompt was: {scene_data['prompt'][:100]}")
            # Fall through to placeholder generation
        
        # Fallback: Create placeholder image with scene text
        logging.info("Generating placeholder image")
        
        img = Image.new('RGB', (512, 768), color=(50, 50, 100))
        draw = ImageDraw.Draw(img)
        
        # Add scene text with sans-serif font
        try:
            # Try common sans-serif fonts in order of preference
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 
                "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
                "/System/Library/Fonts/Helvetica.ttc",  # macOS
                "C:/Windows/Fonts/arial.ttf",  # Windows
            ]
            
            font = None
            for font_path in font_paths:
                try:
                    font = ImageFont.truetype(font_path, 24)
                    logging.info(f"Using font: {font_path}")
                    break
                except:
                    continue
            
            if font is None:
                font = ImageFont.load_default()
                logging.warning("Using default font - sans-serif fonts not found")
        except:
            font = ImageFont.load_default()
        
        # Wrap text
        text = f"Scene {scene_data['scene_id']}\n\n{scene_data['description'][:200]}..."
        
        # Calculate text position
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (512 - text_width) // 2
        y = (768 - text_height) // 2
        
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
        
        # Add frame border
        draw.rectangle([(10, 10), (502, 758)], outline=(255, 255, 255), width=3)
        
        img.save(filepath)
        
        logging.info(f"Generated placeholder image: {filepath}")
        
        # Simulate processing time for consistency
        if not self.use_sd:
            time.sleep(2)
        
        return filepath
    
    def create_video_with_transitions(self, image_data):
        """Create video with transitions between still frames"""
        logging.info("Creating video with transitions...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"generated_video_{timestamp}.mp4"
        video_path = os.path.join(self.videos_dir, video_filename)
        
        # Video settings
        fps = 30
        transition_duration = 1.0  # seconds for transition
        
        # Calculate video dimensions (portrait format)
        width, height = 512, 768
        
        # Initialize video writer with H.264 codec for browser compatibility
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        # Fallback to other codecs if H264 fails
        if not out.isOpened():
            logging.warning("H264 codec failed, trying mp4v...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            
        if not out.isOpened():
            logging.warning("mp4v codec failed, trying XVID...")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_path = video_path.replace('.mp4', '.avi')  # XVID typically uses AVI
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            logging.error("Failed to initialize video writer with any codec!")
            return None
            
        # Detect which codec was used
        if 'H264' in str(fourcc):
            codec_name = "H.264"
        elif 'mp4v' in str(fourcc):
            codec_name = "MPEG-4"
        elif 'XVID' in str(fourcc):
            codec_name = "XVID"
        else:
            codec_name = "Unknown"
            
        logging.info(f"Creating video: {video_path}")
        logging.info(f"Video settings: {width}x{height} @ {fps}fps using {codec_name} codec")
        
        for i, img_data in enumerate(image_data):
            logging.info(f"Processing frame {i+1}/{len(image_data)}")
            
            # Load image
            img = cv2.imread(img_data['path'])
            if img is None:
                logging.error(f"Could not load image: {img_data['path']}")
                continue
            
            img = cv2.resize(img, (width, height))
            
            # Get custom duration for this frame
            frame_duration = img_data.get('duration', 3.0)
            transition_type = img_data.get('transition_type', 'fade')
            
            # Add still frame (hold for duration)
            still_frames = int(frame_duration * fps)
            logging.info(f"Frame {i+1}: holding for {frame_duration}s ({still_frames} frames) with {transition_type} transition")
            for _ in range(still_frames):
                out.write(img)
            
            # Add transition to next frame (except for last frame)
            if i < len(image_data) - 1:
                next_img = cv2.imread(image_data[i + 1]['path'])
                if next_img is not None:
                    next_img = cv2.resize(next_img, (width, height))
                    
                    if transition_type == "cut":
                        # No transition - direct cut
                        logging.info(f"Using cut transition to next frame")
                        pass  # No frames added
                    elif transition_type == "dissolve":
                        # Slower dissolve transition
                        transition_frames = int(transition_duration * 1.5 * fps)  # 50% longer
                        logging.info(f"Using dissolve transition ({transition_frames} frames)")
                        for frame_num in range(transition_frames):
                            alpha = frame_num / transition_frames
                            # Smoother dissolve curve
                            alpha = alpha * alpha * (3.0 - 2.0 * alpha)  # Smoothstep
                            blended = cv2.addWeighted(img, 1 - alpha, next_img, alpha, 0)
                            out.write(blended)
                    else:  # Default fade
                        # Standard fade transition
                        transition_frames = int(transition_duration * fps)
                        logging.info(f"Using fade transition ({transition_frames} frames)")
                        for frame_num in range(transition_frames):
                            alpha = frame_num / transition_frames
                            blended = cv2.addWeighted(img, 1 - alpha, next_img, alpha, 0)
                            out.write(blended)
        
        out.release()
        logging.info(f"Video creation complete: {video_path}")
        
        # If we used H264, the video should be browser-compatible
        # If we used mp4v or XVID, try to convert with ffmpeg for better compatibility
        if codec_name != "H.264" and video_path.endswith('.mp4'):
            logging.info("Attempting to convert video to H.264 for better browser compatibility...")
            h264_path = video_path.replace('.mp4', '_h264.mp4')
            
            try:
                import subprocess
                result = subprocess.run([
                    'ffmpeg', '-i', video_path, '-c:v', 'libx264', 
                    '-preset', 'fast', '-crf', '23', '-y', h264_path
                ], capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    logging.info(f"Successfully converted to H.264: {h264_path}")
                    os.remove(video_path)  # Remove original
                    return h264_path
                else:
                    logging.warning(f"ffmpeg conversion failed: {result.stderr}")
                    
            except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
                logging.warning(f"Could not convert with ffmpeg: {str(e)}")
        
        return video_path