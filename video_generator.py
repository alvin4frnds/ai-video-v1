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
        
        # Initialize Mixtral client
        self.mixtral = MixtralClient()
        
        # Initialize Stable Diffusion client
        self.sd_client = StableDiffusionClient()
        
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
            self.use_sd = True
        else:
            logging.warning("Stable Diffusion WebUI unavailable - will use placeholder images")
            self.use_sd = False
    
    def analyze_prompt(self, prompt):
        """Analyze text prompt and extract scenes/narrative elements using Mixtral"""
        logging.info("Analyzing prompt for narrative structure with Mixtral LLM...")
        
        # Use Mixtral for intelligent scene analysis
        scenes = self.mixtral.analyze_narrative_structure(prompt)
        
        logging.info(f"Extracted {len(scenes)} scenes from prompt")
        for i, scene in enumerate(scenes):
            logging.info(f"Scene {i+1}: {scene[:100]}...")
        
        return scenes
    
    def plan_sequences(self, scenes):
        """Plan the sequence of still frames needed using Mixtral-enhanced prompts"""
        logging.info("Planning frame sequences with Mixtral-enhanced prompts...")
        
        # Use Mixtral to generate enhanced prompts for each scene
        enhanced_scenes = self.mixtral.generate_scene_prompts(scenes)
        
        scene_plan = []
        for enhanced_scene in enhanced_scenes:
            frame_data = {
                'scene_id': enhanced_scene['scene_id'],
                'description': enhanced_scene['original_description'],
                'prompt': enhanced_scene['enhanced_prompt'],
                'duration': enhanced_scene['duration'],
                'transition_type': enhanced_scene['transition_type']
            }
            scene_plan.append(frame_data)
            
            logging.info(f"Planned frame {enhanced_scene['scene_id']}: {frame_data['prompt'][:80]}...")
        
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
            
            # Generate with SD
            result_path = self.sd_client.generate_image(
                prompt=scene_data['prompt'],
                output_path=filepath,
                width=1024,
                height=576,
                steps=25,  # Faster generation for video frames
                cfg_scale=7.5
            )
            
            if result_path:
                logging.info(f"SD image generated successfully: {filepath}")
                return result_path
            else:
                logging.warning("SD generation failed, falling back to placeholder")
                # Fall through to placeholder generation
        
        # Fallback: Create placeholder image with scene text
        logging.info("Generating placeholder image")
        
        img = Image.new('RGB', (1024, 576), color=(50, 50, 100))
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
        x = (1024 - text_width) // 2
        y = (576 - text_height) // 2
        
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
        
        # Add frame border
        draw.rectangle([(10, 10), (1014, 566)], outline=(255, 255, 255), width=3)
        
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
        frame_duration = 3.0  # seconds per still
        transition_duration = 1.0  # seconds for transition
        
        # Calculate video dimensions
        width, height = 1024, 576
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        logging.info(f"Creating video: {video_path}")
        logging.info(f"Video settings: {width}x{height} @ {fps}fps")
        
        for i, img_data in enumerate(image_data):
            logging.info(f"Processing frame {i+1}/{len(image_data)}")
            
            # Load image
            img = cv2.imread(img_data['path'])
            if img is None:
                logging.error(f"Could not load image: {img_data['path']}")
                continue
            
            img = cv2.resize(img, (width, height))
            
            # Add still frame (hold for duration)
            still_frames = int(frame_duration * fps)
            for _ in range(still_frames):
                out.write(img)
            
            # Add transition to next frame (except for last frame)
            if i < len(image_data) - 1:
                next_img = cv2.imread(image_data[i + 1]['path'])
                if next_img is not None:
                    next_img = cv2.resize(next_img, (width, height))
                    
                    # Create fade transition
                    transition_frames = int(transition_duration * fps)
                    for frame_num in range(transition_frames):
                        alpha = frame_num / transition_frames
                        blended = cv2.addWeighted(img, 1 - alpha, next_img, alpha, 0)
                        out.write(blended)
        
        out.release()
        logging.info(f"Video creation complete: {video_path}")
        
        return video_path