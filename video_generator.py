import os
import re
import logging
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
from datetime import datetime
from typing import List, Optional
from mixtral_client import MixtralClient
from sd_client import StableDiffusionClient
from face_analyzer import FaceAnalyzer
from face_selector import FaceSelector

class VideoGenerator:
    def __init__(self, candidate_count=6):
        self.output_dir = "output"
        self.frames_dir = os.path.join(self.output_dir, "frames")
        self.videos_dir = os.path.join(self.output_dir, "videos")
        self.candidate_count = candidate_count
        
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.videos_dir, exist_ok=True)
        
        # Generate YmdH format seed for consistent results within the hour
        self.base_seed = int(datetime.now().strftime("%Y%m%d%H"))
        logging.info(f"Using base seed: {self.base_seed} (YmdH format for consistency)")
        logging.info(f"Candidate images per scene: {self.candidate_count}")
        
        # Initialize Mixtral client
        self.mixtral = MixtralClient()
        self.mixtral.base_seed = self.base_seed  # Share seed for consistency
        
        # Initialize Stable Diffusion client with network IP
        self.sd_client = StableDiffusionClient(base_url="http://192.168.0.199:8001")
        
        # Initialize Face Analyzer and Face Selector
        self.face_analyzer = FaceAnalyzer()
        self.face_selector = FaceSelector()
        
        # Select random face for consistency
        self.selected_face_path = self.face_selector.select_random_face(seed=self.base_seed)
        if self.selected_face_path:
            face_info = self.face_selector.get_current_selection()
            logging.info(f"🎭 Selected face for video: {face_info['face_filename']} from {face_info['person']}")
            logging.info(f"🎭 Face will be swapped into all generated images for ultimate consistency")
        else:
            logging.warning("⚠️  No face images found - will use character consistency only")
        
        logging.info("VideoGenerator initialized")
        logging.info(f"Output directory: {self.output_dir}")
        
        # Log available faces
        available_faces = self.face_selector.list_all_faces()
        if available_faces:
            logging.info(f"👥 Available faces: {dict(available_faces)}")
        else:
            logging.info("👥 No face images found in in/individual/ directory")
        
        # Check Mixtral availability
        if self.mixtral.check_connection():
            logging.info("Mixtral LLM connected and ready for prompt enhancement")
        else:
            logging.warning("Mixtral LLM unavailable - will use fallback prompt generation")
        
        # Check Stable Diffusion availability
        if self.sd_client.check_connection():
            current_model = self.sd_client.get_current_model()
            logging.info(f"Stable Diffusion WebUI connected - Model: {current_model}")
            
            # Now that SD is connected, check for ADetailer models
            self.sd_client._check_adetailer_models()
            
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
                # Clean up test file if it exists
                if os.path.exists(test_path):
                    os.remove(test_path)
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
        
        logging.info(f"🎭 Generating consistent character profile for {len(scene_descriptions)} scenes...")
        
        # Use Mixtral to generate enhanced prompts for each scene with character consistency
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
        """Generate image for a scene using batch generation with face analysis"""
        scene_id = scene_data['scene_id']
        logging.info(f"🎬 Starting image generation for Scene {scene_id}")
        logging.info(f"📝 Scene description: {scene_data['description'][:100]}{'...' if len(scene_data['description']) > 100 else ''}")
        
        # Initialize batch analysis data
        batch_analysis = []
        
        # Log face selection info
        if self.selected_face_path:
            face_info = self.face_selector.get_current_selection()
            logging.info(f"🎭 Scene {scene_id} will use face: {face_info['face_filename']} from {face_info['person']}")
            logging.info(f"🎭 Roop available: {getattr(self.sd_client, 'roop_available', False)}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_filename = f"frame_{scene_id:03d}_{timestamp}.png"
        final_filepath = os.path.join(self.frames_dir, final_filename)
        
        if self.use_sd:
            # Create batch directory for this scene
            batch_dir = os.path.join(self.frames_dir, f"batch_scene_{scene_id}")
            os.makedirs(batch_dir, exist_ok=True)
            
            logging.info(f"🤖 Using SD WebUI with batch generation and face analysis")
            
            # Generate batch with consistent seed
            scene_seed = self.base_seed + scene_id
            logging.info(f"🎲 Scene {scene_id} seed: {scene_seed} (base: {self.base_seed} + scene: {scene_id})")
            logging.info(f"📁 Batch directory: {os.path.basename(batch_dir)}")
            
            # Log negative prompt usage and character consistency
            negative_prompt = scene_data.get('negative_prompt', '')
            if negative_prompt:
                logging.info(f"🚫 Character consistency enforced: {negative_prompt[:60]}...")
                if 'different clothing' in negative_prompt or 'different appearance' in negative_prompt:
                    logging.info(f"🎭 Character variation prevention active")
            else:
                logging.info(f"🚫 Using default character consistency constraints")
            
            # Check if we should use Roop face swapping
            if self.selected_face_path and getattr(self.sd_client, 'roop_available', False):
                logging.info(f"🎭 Using Roop face swapping for ultimate consistency")
                logging.info(f"🎭 Generating {self.candidate_count} images with face swap from {os.path.basename(self.selected_face_path)}")
                
                batch_paths = self.sd_client.generate_batch_with_face_swap(
                    prompt=scene_data['prompt'],
                    face_image_path=self.selected_face_path,
                    output_dir=batch_dir,
                    count=self.candidate_count,
                    width=512,
                    height=768,
                    steps=28,
                    cfg_scale=7,
                    sampler="DPM++ 2M SDE",
                    scheduler="Karras",
                    seed=scene_seed,
                    negative_prompt=scene_data.get('negative_prompt', '')
                )
            else:
                # Fallback to regular batch generation
                if self.selected_face_path:
                    logging.info(f"⚠️  Roop not available - using regular generation with character consistency")
                
                batch_paths = self.sd_client.generate_batch_images(
                    prompt=scene_data['prompt'],
                    output_dir=batch_dir,
                    count=self.candidate_count,
                    width=512,
                    height=768,
                    steps=28,
                    cfg_scale=7,
                    sampler="DPM++ 2M SDE",
                    scheduler="Karras",
                    seed=scene_seed,
                    negative_prompt=scene_data.get('negative_prompt', '')
                )
            
            if batch_paths:
                logging.info(f"🔎 Starting face analysis and selection for {len(batch_paths)} images...")
                
                # Analyze faces in all generated images
                # Get reference images for clothing consistency (previous selected frames)
                reference_images = []
                if scene_id > 1:  # If not first scene, use previous frames as reference
                    for ref_scene_id in range(1, scene_id):
                        ref_frame_pattern = os.path.join(self.frames_dir, f"frame_{ref_scene_id:03d}_*.png")
                        import glob
                        ref_files = glob.glob(ref_frame_pattern)
                        if ref_files:
                            reference_images.append(ref_files[0])  # Use most recent
                
                best_image, batch_analysis = self._select_best_image(batch_paths, scene_data, reference_images)
                
                if best_image:
                    # Copy best image to final location
                    import shutil
                    logging.info(f"📋 Copying selected image to final location...")
                    shutil.copy2(best_image, final_filepath)
                    
                    best_filename = os.path.basename(best_image)
                    logging.info(f"✅ Scene {scene_id} complete: {best_filename} → {final_filename}")
                    logging.info(f"💾 Final image saved: {final_filepath}")
                    
                    # Clean up batch directory (optional - keep for debugging)
                    # shutil.rmtree(batch_dir)
                    
                    return {
                        'final_path': final_filepath,
                        'batch_analysis': batch_analysis,
                        'scene_id': scene_id
                    }
                else:
                    logging.warning("⚠️  Face analysis failed to select best image, using first available")
                    if batch_paths:
                        import shutil
                        shutil.copy2(batch_paths[0], final_filepath)
                        logging.info(f"📄 Using fallback image: {os.path.basename(batch_paths[0])} → {final_filename}")
                        return {
                            'final_path': final_filepath,
                            'batch_analysis': batch_analysis,
                            'scene_id': scene_id
                        }
            
            logging.warning(f"💥 Batch generation failed for Scene {scene_id}, falling back to placeholder")
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
        
        img.save(final_filepath)
        
        logging.info(f"Generated placeholder image: {final_filepath}")
        
        # Simulate processing time for consistency
        if not self.use_sd:
            time.sleep(2)
        
        return {
            'final_path': final_filepath,
            'batch_analysis': batch_analysis,
            'scene_id': scene_id
        }
    
    def _select_best_image(self, image_paths: List[str], scene_data: dict, reference_images: List[str] = None) -> tuple[Optional[str], List[dict]]:
        """Select the best image from batch based on face analysis, clothing consistency, and return analysis data"""
        if not image_paths:
            return None, []
        
        logging.info(f"🔍 Analyzing {len(image_paths)} generated images for quality and face realism...")
        
        # Analyze faces in all images
        image_analysis = []
        for i, img_path in enumerate(image_paths):
            filename = os.path.basename(img_path)
            logging.info(f"👤 Analyzing faces in image {i+1}/{len(image_paths)}: {filename}")
            
            face_data = self.face_analyzer.detect_faces(img_path)
            
            # Analyze clothing consistency if reference images available
            clothing_data = {}
            if reference_images:
                clothing_data = self.face_analyzer.analyze_clothing_consistency(img_path, reference_images)
                logging.info(f"   👕 Clothing consistency: {clothing_data.get('consistency_score', 0):.2f}")
            
            analysis = {
                'path': img_path,
                'face_data': face_data,
                'clothing_data': clothing_data,
                'filename': filename
            }
            image_analysis.append(analysis)
            
            # Detailed face analysis results
            face_count = face_data.get('face_count', 0)
            quality_score = face_data.get('quality_score', 0)
            has_realistic = face_data.get('has_realistic_face', False)
            methods = face_data.get('detection_methods', [])
            
            if face_count > 0:
                logging.info(f"   ✅ Found {face_count} face(s), quality: {quality_score:.2f}, realistic: {has_realistic}")
                logging.info(f"   🔬 Detection methods: {', '.join(methods) if methods else 'None'}")
                
                # Log individual face details
                for j, face in enumerate(face_data.get('faces', [])):
                    conf = face.get('confidence', 0)
                    method = face.get('method', 'Unknown')
                    area = face.get('area', 0)
                    logging.info(f"      Face {j+1}: {method} detection, confidence: {conf:.2f}, area: {area}px")
            else:
                error_msg = face_data.get('error', 'Unknown issue')
                logging.info(f"   ❌ No faces detected - {error_msg}")
            
            # Log clothing consistency results
            if clothing_data:
                consistency = clothing_data.get('consistency_score', 0)
                colors = clothing_data.get('dominant_colors', [])
                if consistency > 0:
                    logging.info(f"   👕 Clothing consistency: {consistency:.2f}, colors: {colors[:2] if colors else 'N/A'}")
                if clothing_data.get('error'):
                    logging.info(f"   ⚠️ Clothing analysis error: {clothing_data['error']}")
        
        logging.info(f"📊 Scoring all {len(image_analysis)} images...")
        
        # Score each image
        scored_images = []
        for i, analysis in enumerate(image_analysis):
            score = self._calculate_image_score(analysis, scene_data)
            scored_images.append((score, analysis))
            
            # Enhanced logging with breakdown
            face_qual = analysis['face_data'].get('quality_score', 0)
            clothing_cons = analysis.get('clothing_data', {}).get('consistency_score', 0.5)
            logging.info(f"📈 Image {i+1} ({analysis['filename']}): overall score {score:.3f} (face: {face_qual:.2f}, clothing: {clothing_cons:.2f})")
        
        # Sort by score (highest first)
        scored_images.sort(key=lambda x: x[0], reverse=True)
        
        if scored_images:
            # Log ranking
            logging.info(f"🏆 Image ranking (best to worst):")
            for i, (score, analysis) in enumerate(scored_images):
                rank_icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
                face_qual = analysis['face_data'].get('quality_score', 0)
                clothing_cons = analysis.get('clothing_data', {}).get('consistency_score', 0.5)
                logging.info(f"   {rank_icon} {analysis['filename']}: {score:.3f} (F:{face_qual:.2f} C:{clothing_cons:.2f})")
            
            best_score, best_analysis = scored_images[0]
            best_face = best_analysis['face_data'].get('quality_score', 0)
            best_clothing = best_analysis.get('clothing_data', {}).get('consistency_score', 0.5)
            logging.info(f"🎯 Selected best image: {best_analysis['filename']} with score {best_score:.3f} (face: {best_face:.2f}, clothing: {best_clothing:.2f})")
            
            # Prepare batch analysis data for UI
            batch_analysis = []
            for score, analysis in scored_images:
                face_data = analysis['face_data']
                batch_analysis.append({
                    'path': analysis['path'],
                    'filename': analysis['filename'],
                    'score': score,
                    'face_count': face_data.get('face_count', 0),
                    'quality_score': face_data.get('quality_score', 0),
                    'has_realistic_face': face_data.get('has_realistic_face', False),
                    'detection_methods': face_data.get('detection_methods', []),
                    'faces': face_data.get('faces', []),
                    'error': face_data.get('error', None),
                    'clothing_consistency': analysis.get('clothing_data', {}).get('consistency_score', 0.5),
                    'clothing_colors': analysis.get('clothing_data', {}).get('dominant_colors', []),
                    'clothing_error': analysis.get('clothing_data', {}).get('error', None),
                    'roop_face_used': self.selected_face_path is not None,
                    'selected_face_person': self.face_selector.selected_person if self.selected_face_path else None,
                    'is_selected': analysis == best_analysis
                })
            
            return best_analysis['path'], batch_analysis
        
        logging.warning("⚠️  No images could be scored properly")
        return None, []
    
    def _calculate_image_score(self, analysis: dict, scene_data: dict) -> float:
        """Calculate overall score for an image including face quality and clothing consistency"""
        face_data = analysis['face_data']
        clothing_data = analysis.get('clothing_data', {})
        
        # Base quality score from face analysis
        face_quality = face_data.get('quality_score', 0.0)
        
        # Face count score
        face_count = face_data.get('face_count', 0)
        if 'person' in scene_data.get('description', '').lower() or 'woman' in scene_data.get('description', '').lower():
            # Scenes with people should have faces
            if face_count >= 1:
                face_count_score = 1.0
            else:
                face_count_score = 0.3  # Penalty for missing faces
        else:
            # Scenes without people mentioned
            face_count_score = 0.8  # Neutral score
        
        # Face realism score
        face_realism = 1.0 if face_data.get('has_realistic_face', False) else 0.5
        
        # Detection method score (prefer MediaPipe over OpenCV)
        methods = face_data.get('detection_methods', [])
        if 'MediaPipe' in methods:
            detection_score = 1.0
        elif 'OpenCV' in methods:
            detection_score = 0.8
        else:
            detection_score = 0.6
        
        # Clothing consistency score (NEW)
        clothing_score = clothing_data.get('consistency_score', 0.5)  # Default neutral if no reference
        
        # Error penalties
        face_error_penalty = 0.0 if 'error' not in face_data or face_data['error'] is None else 0.3
        clothing_error_penalty = 0.0 if 'error' not in clothing_data or clothing_data['error'] is None else 0.1
        
        # Combined score with clothing consistency as a major factor
        total_score = (
            face_quality * 0.25 +           # Face detection quality
            face_count_score * 0.20 +       # Correct number of faces
            face_realism * 0.20 +           # Face realism
            detection_score * 0.10 +        # Detection method reliability
            clothing_score * 0.25           # Character consistency (NEW - high weight)
        ) - face_error_penalty - clothing_error_penalty
        
        return max(0.0, min(1.0, total_score))
    
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
        # Use 'avc1' instead of 'H264' for better MP4 container compatibility
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        # Fallback to H264 tag if avc1 fails
        if not out.isOpened():
            logging.info("avc1 codec failed, trying H264 tag...")
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        # Fallback to mp4v if H264 fails
        if not out.isOpened():
            logging.warning("H264 codec failed, trying mp4v...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            
        # Fallback to XVID as last resort
        if not out.isOpened():
            logging.warning("mp4v codec failed, trying XVID...")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_path = video_path.replace('.mp4', '.avi')  # XVID typically uses AVI
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            logging.error("Failed to initialize video writer with any codec!")
            return None
            
        # Detect which codec was used
        fourcc_str = ''.join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        if fourcc_str in ['avc1', 'H264']:
            codec_name = "H.264"
        elif fourcc_str == 'mp4v':
            codec_name = "MPEG-4"
        elif fourcc_str == 'XVID':
            codec_name = "XVID"
        else:
            codec_name = f"Unknown ({fourcc_str})"
            
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