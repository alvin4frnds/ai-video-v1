#!/usr/bin/env python3
"""
Simple WAN-style Video Generator
Creates smooth transitions between two images without heavy diffusion models
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2
from pathlib import Path
import logging
from typing import List
import math

# Setup logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional scipy import with fallback
try:
    from scipy.ndimage import map_coordinates
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available - using fallback motion interpolation")

class SimpleWANGenerator:
    def __init__(self):
        self.output_dir = Path("output/videos")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_motion_field(self, start: np.ndarray, end: np.ndarray, context: str = "general") -> np.ndarray:
        """Create motion field for realistic subject movement"""
        h, w = start.shape[:2]
        motion_field = np.zeros((h, w, 2), dtype=np.float32)
        
        # Create natural movement patterns based on context
        center_x, center_y = w//2, h//3  # Focus on upper body/face area
        
        for y in range(h):
            for x in range(w):
                # Distance from center for natural falloff
                dx = x - center_x
                dy = y - center_y
                distance = math.sqrt(dx*dx + dy*dy)
                max_distance = math.sqrt(center_x*center_x + center_y*center_y)
                
                # Normalize distance (0 = center, 1 = edge)
                norm_dist = min(distance / max_distance, 1.0)
                
                # Create subtle organic movement
                # More movement in center (subject), less at edges (background)
                movement_strength = (1.0 - norm_dist) * 0.8
                
                # Add natural variation based on position
                movement_x = movement_strength * 3 * math.sin(y / h * math.pi)
                movement_y = movement_strength * 2 * math.cos(x / w * math.pi)
                
                motion_field[y, x, 0] = movement_x
                motion_field[y, x, 1] = movement_y
        
        return motion_field
    
    def warp_image(self, image: np.ndarray, motion_field: np.ndarray, t: float) -> np.ndarray:
        """Warp image to simulate subject movement"""
        if not SCIPY_AVAILABLE:
            # Fallback: simple image translation
            h, w = image.shape[:2]
            shift_x = int(np.mean(motion_field[:, :, 0]) * t)
            shift_y = int(np.mean(motion_field[:, :, 1]) * t)
            
            if abs(shift_x) > 0 or abs(shift_y) > 0:
                M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            return image
        
        h, w = image.shape[:2]
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # Apply motion field scaled by t
        x_coords_new = x_coords + motion_field[:, :, 0] * t
        y_coords_new = y_coords + motion_field[:, :, 1] * t
        
        # Ensure coordinates are within bounds
        x_coords_new = np.clip(x_coords_new, 0, w-1)
        y_coords_new = np.clip(y_coords_new, 0, h-1)
        
        # Warp each channel separately
        warped = np.zeros_like(image)
        for c in range(image.shape[2]):
            warped[:, :, c] = map_coordinates(
                image[:, :, c], 
                [y_coords_new, x_coords_new], 
                order=1, 
                mode='nearest'
            )
        
        return warped
    
    def smooth_interpolation(self, start_img: np.ndarray, end_img: np.ndarray, num_frames: int = 81) -> List[np.ndarray]:
        """Create smooth interpolation with realistic subject movement"""
        logger.info(f"Creating {num_frames} interpolated frames with motion...")
        
        # Create motion field for realistic movement
        motion_field = self.create_motion_field(start_img, end_img)
        
        frames = []
        
        for i in range(num_frames):
            # Use smooth easing function instead of linear
            t = i / (num_frames - 1)
            
            # Apply smooth step easing (3t² - 2t³)
            smooth_t = 3 * t * t - 2 * t * t * t
            
            # Add organic variation for natural movement
            noise_factor = 0.015 * math.sin(t * math.pi * 4)  # Subtle oscillation
            final_t = smooth_t + noise_factor
            final_t = max(0, min(1, final_t))  # Clamp to [0,1]
            
            try:
                # Create warped versions for realistic motion
                start_warped = self.warp_image(start_img, motion_field, -final_t * 0.4)
                end_warped = self.warp_image(end_img, motion_field, (1-final_t) * 0.4)
                
                # Blend warped images
                interpolated = start_warped * (1 - final_t) + end_warped * final_t
                
            except Exception as e:
                logger.warning(f"Motion interpolation failed at frame {i}: {e}")
                # Fallback to simple interpolation
                interpolated = start_img * (1 - final_t) + end_img * final_t
            
            frames.append(interpolated.astype(np.uint8))
            
            if i % 10 == 0:
                logger.info(f"Generated frame {i+1}/{num_frames} with motion")
        
        return frames
    
    def add_motion_blur(self, frame: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """Add subtle motion blur for more realistic movement"""
        if intensity <= 0:
            return frame
            
        # Create motion blur kernel
        kernel_size = max(3, int(intensity * 5))
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size//2, :] = 1.0
        kernel = kernel / kernel_size
        
        # Apply blur
        blurred = cv2.filter2D(frame, -1, kernel)
        
        # Blend with original
        return (frame * (1 - intensity * 0.3) + blurred * (intensity * 0.3)).astype(np.uint8)
    
    def enhance_transition(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Apply enhancements to make the transition more natural"""
        logger.info("Enhancing transition with effects...")
        
        enhanced_frames = []
        total_frames = len(frames)
        
        for i, frame in enumerate(frames):
            # Calculate position in transition (0 to 1)
            t = i / (total_frames - 1)
            
            # Add subtle motion blur in middle of transition
            blur_intensity = 4 * t * (1 - t)  # Peaks at t=0.5
            enhanced_frame = self.add_motion_blur(frame, blur_intensity * 0.3)
            
            # Subtle brightness variation for more dynamic feel
            brightness_factor = 1.0 + 0.05 * math.sin(t * math.pi)
            enhanced_frame = np.clip(enhanced_frame * brightness_factor, 0, 255).astype(np.uint8)
            
            enhanced_frames.append(enhanced_frame)
        
        return enhanced_frames
    
    def save_video(self, frames: List[np.ndarray], output_path: str, fps: int = 25):
        """Save frames as MP4 video"""
        logger.info(f"Saving video to {output_path}")
        
        if not frames:
            raise ValueError("No frames to save")
        
        height, width = frames[0].shape[:2]
        
        # Use H.264 codec for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for i, frame in enumerate(frames):
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            
            if i % 20 == 0:
                logger.info(f"Writing frame {i+1}/{len(frames)}")
        
        out.release()
        logger.info(f"✅ Video saved: {output_path}")
    
    def generate_video(self, 
                      start_image_path: str,
                      end_image_path: str,
                      output_path: str,
                      num_frames: int = 81,
                      fps: int = 25,
                      target_size: tuple = (480, 768)):
        """Generate transition video between two images"""
        
        logger.info("🎬 Starting Simple WAN-style video generation...")
        logger.info(f"Start image: {start_image_path}")
        logger.info(f"End image: {end_image_path}")
        logger.info(f"Output: {output_path}")
        logger.info(f"Frames: {num_frames}, FPS: {fps}")
        
        # Load and prepare images
        start_img = Image.open(start_image_path).convert('RGB')
        end_img = Image.open(end_image_path).convert('RGB')
        
        logger.info(f"Original sizes - Start: {start_img.size}, End: {end_img.size}")
        
        # Resize to target size
        start_img = start_img.resize(target_size)
        end_img = end_img.resize(target_size)
        
        # Convert to numpy arrays
        start_array = np.array(start_img)
        end_array = np.array(end_img)
        
        logger.info(f"Resized to: {target_size}")
        
        # Generate interpolated frames
        frames = self.smooth_interpolation(start_array, end_array, num_frames)
        
        # Enhance transition
        enhanced_frames = self.enhance_transition(frames)
        
        # Save video
        self.save_video(enhanced_frames, output_path, fps)
        
        return output_path

def run_stage_generation():
    """Generate videos for all stages"""
    generator = SimpleWANGenerator()
    
    # Define stages with their prompts (for reference)
    stages = [
        {
            "name": "Stage 1: Park Walk Pose Transition", 
            "start": "in/comfy/1_start.jpeg",
            "end": "in/comfy/1_end.jpeg",
            "prompt": "Smooth pose transition in park setting"
        },
        {
            "name": "Stage 2: Walking Motion",
            "start": "in/comfy/2_start.jpeg", 
            "end": "in/comfy/2_end.jpeg",
            "prompt": "Standing to walking motion transition"
        },
        {
            "name": "Stage 3: Expression Change",
            "start": "in/comfy/3_start.jpeg",
            "end": "in/comfy/3_end.jpeg", 
            "prompt": "Neutral to smile expression transition"
        },
        {
            "name": "Stage 4: Pose Variation",
            "start": "in/comfy/4_start.jpeg",
            "end": "in/comfy/4_end.jpeg",
            "prompt": "Arms crossed to arms at sides"
        },
        {
            "name": "Stage 5: Action Sequence", 
            "start": "in/comfy/5_start.jpeg",
            "end": "in/comfy/5_end.jpeg",
            "prompt": "Sitting to standing motion"
        },
        {
            "name": "Stage 6: Finale",
            "start": "in/comfy/6_start.jpeg",
            "end": "in/comfy/6_end.jpeg",
            "prompt": "Looking down to looking up transition"
        }
    ]
    
    # Also create the basic test video
    basic_stages = [
        {
            "name": "Basic Test: Blue to Red Gradient",
            "start": "in/comfy/1.jpeg",
            "end": "in/comfy/2.jpeg", 
            "prompt": "Simple color transition test"
        }
    ]
    
    all_stages = basic_stages + stages
    generated_videos = []
    
    for i, stage in enumerate(all_stages):
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"🎭 {stage['name']}")
            logger.info(f"📝 {stage['prompt']}")
            logger.info(f"{'='*60}")
            
            # Check if source images exist
            if not Path(stage["start"]).exists():
                logger.warning(f"Start image not found: {stage['start']}")
                continue
                
            if not Path(stage["end"]).exists():
                logger.warning(f"End image not found: {stage['end']}")
                continue
            
            # Generate output filename
            output_filename = f"wan_stage_{i+1:02d}_transition.mp4"
            output_path = f"output/videos/{output_filename}"
            
            # Generate video
            result_path = generator.generate_video(
                start_image_path=stage["start"],
                end_image_path=stage["end"], 
                output_path=output_path,
                num_frames=81,
                fps=25
            )
            
            generated_videos.append({
                "name": stage["name"],
                "path": result_path,
                "prompt": stage["prompt"]
            })
            
            logger.info(f"✅ Successfully generated: {result_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate {stage['name']}: {e}")
            continue
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"🎉 GENERATION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Generated {len(generated_videos)} videos:")
    
    for video in generated_videos:
        logger.info(f"  ✅ {video['name']}")
        logger.info(f"     📁 {video['path']}")
        logger.info(f"     📝 {video['prompt']}")
        print()

if __name__ == "__main__":
    run_stage_generation()