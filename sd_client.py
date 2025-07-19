import requests
import json
import logging
import base64
from io import BytesIO
from PIL import Image
import os
from datetime import datetime
from typing import Dict, Optional, List

class StableDiffusionClient:
    """Client for interacting with Automatic1111 Stable Diffusion WebUI"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.session = requests.Session()
        self.available_adetailer_models = []
        self.roop_available = False
        logging.info(f"Initialized Stable Diffusion client with base URL: {base_url}")
        
        # Check for extensions availability
        self._check_adetailer_models()
        self._check_roop_availability()
    
    def _check_adetailer_models(self):
        """Check which ADetailer models are available"""
        self.preferred_models = [
            "face_yolov8n.pt",
            "hand_yolov8n.pt", 
            "person_yolov8n-seg.pt",
            "yolov8x-worldv2.pt",
            "mediapipe_face_full",
            "mediapipe_face_mesh",
            "mediapipe_face_mesh_eyes_only"
        ]
        
        try:
            # Try to get ADetailer models from SD WebUI
            response = self.session.get(f"{self.base_url}/adetailer/v1/ad_model", timeout=5)
            if response.status_code == 200:
                result = response.json()
                available_models = result.get("ad_model", [])
                self.available_adetailer_models = [model for model in self.preferred_models if model in available_models]
                if self.available_adetailer_models:
                    logging.info(f"🔧 ADetailer models available: {', '.join(self.available_adetailer_models)}")
                else:
                    logging.warning("⚠️  ADetailer extension found but no preferred models available")
                    logging.debug(f"Available models: {available_models}")
                    logging.debug(f"Preferred models: {self.preferred_models}")
            else:
                logging.warning("⚠️  ADetailer extension not found or not responding")
        except Exception as e:
            logging.info(f"ℹ️  ADetailer model check skipped: {str(e)}")
    
    def get_adetailer_config(self, model_type: str = "comprehensive") -> List[Dict]:
        """Get ADetailer configuration based on available models"""
        configs = []
        
        if not self.available_adetailer_models:
            return configs
        
        # Face detection configuration
        if "face_yolov8n.pt" in self.available_adetailer_models:
            configs.append({
                "ad_model": "face_yolov8n.pt",
                "ad_prompt": "single person, perfect face, detailed eyes, clear skin, natural expression, high quality, one person only",
                "ad_negative_prompt": "blurry face, distorted features, bad eyes, deformed face, low quality, multiple people, crowd, group, two people, extra people",
                "ad_confidence": 0.3,
                "ad_mask_merge_invert": "None",
                "ad_mask_blur": 4,
                "ad_denoising_strength": 0.4,
                "ad_inpaint_only_masked": True,
                "ad_inpaint_only_masked_padding": 32,
                "ad_use_inpaint_width_height": False,
                "ad_steps": 20,
                "ad_cfg_scale": 7.0,
                "ad_checkpoint": "Use same checkpoint",
                "ad_vae": "Use same VAE",
                "ad_sampler": "DPM++ 2M SDE",
                "ad_scheduler": "Karras"
            })
        elif "mediapipe_face_full" in self.available_adetailer_models:
            configs.append({
                "ad_model": "mediapipe_face_full",
                "ad_prompt": "single person, perfect face, detailed eyes, clear skin, natural expression, one person only",
                "ad_negative_prompt": "blurry face, distorted features, bad eyes, deformed face, multiple people, crowd, group, two people",
                "ad_confidence": 0.5,
                "ad_mask_blur": 4,
                "ad_denoising_strength": 0.35,
                "ad_inpaint_only_masked": True,
                "ad_inpaint_only_masked_padding": 32,
                "ad_steps": 20,
                "ad_cfg_scale": 7.0,
                "ad_sampler": "DPM++ 2M SDE"
            })
        
        # Hand detection configuration
        if "hand_yolov8n.pt" in self.available_adetailer_models:
            configs.append({
                "ad_model": "hand_yolov8n.pt",
                "ad_prompt": "single person hands, perfect hands, detailed fingers, natural pose, correct anatomy, one person only",
                "ad_negative_prompt": "deformed hands, bad fingers, extra fingers, missing fingers, blurry hands, multiple people, other person hands",
                "ad_confidence": 0.3,
                "ad_mask_blur": 4,
                "ad_denoising_strength": 0.5,
                "ad_inpaint_only_masked": True,
                "ad_inpaint_only_masked_padding": 32,
                "ad_steps": 20,
                "ad_cfg_scale": 7.0,
                "ad_sampler": "DPM++ 2M SDE"
            })
        
        # Person segmentation configuration
        if "person_yolov8n-seg.pt" in self.available_adetailer_models:
            configs.append({
                "ad_model": "person_yolov8n-seg.pt",
                "ad_prompt": "single person, high quality person, detailed features, natural proportions, one person only",
                "ad_negative_prompt": "low quality, distorted proportions, blurry, multiple people, crowd, group, two people, additional person",
                "ad_confidence": 0.3,
                "ad_mask_blur": 4,
                "ad_denoising_strength": 0.3,
                "ad_inpaint_only_masked": True,
                "ad_inpaint_only_masked_padding": 32,
                "ad_steps": 15,
                "ad_cfg_scale": 6.0,
                "ad_sampler": "DPM++ 2M SDE"
            })
        elif "yolov8x-worldv2.pt" in self.available_adetailer_models:
            configs.append({
                "ad_model": "yolov8x-worldv2.pt",
                "ad_prompt": "single person, high quality, detailed features, one person only",
                "ad_negative_prompt": "low quality, blurry, multiple people, crowd, group, two people",
                "ad_confidence": 0.3,
                "ad_mask_blur": 4,
                "ad_denoising_strength": 0.25,
                "ad_inpaint_only_masked": True,
                "ad_inpaint_only_masked_padding": 32,
                "ad_steps": 15,
                "ad_cfg_scale": 6.0,
                "ad_sampler": "DPM++ 2M SDE"
            })
        
        return configs
    
    def _get_adetailer_scripts(self) -> Dict:
        """Get ADetailer scripts configuration"""
        adetailer_configs = self.get_adetailer_config()
        
        if not adetailer_configs:
            # Return empty scripts if no ADetailer models available
            return {}
        
        # Build the script args with available models
        script_args = [
            True,   # Enable ADetailer
            False,  # Skip img2img
        ]
        script_args.extend(adetailer_configs)
        
        model_names = [config.get('ad_model', 'unknown') for config in adetailer_configs]
        logging.debug(f"🔧 ADetailer configured with models: {', '.join(model_names)}")
        
        return {
            "ADetailer": {
                "args": script_args
            }
        }
    
    def find_sd_webui(self) -> Optional[str]:
        """Try to find SD WebUI on common ports and IPs"""
        # Common ports to check
        common_ports = [8001, 7860, 7861, 8000, 8080]
        # Common IPs to check (localhost and common network IPs)
        common_ips = ["localhost", "127.0.0.1", "192.168.0.199"]
        
        logging.info("🔍 Searching for SD WebUI on common ports and IPs...")
        
        for ip in common_ips:
            for port in common_ports:
                test_url = f"http://{ip}:{port}"
                try:
                    response = self.session.get(f"{test_url}/sdapi/v1/options", timeout=3)
                    if response.status_code == 200:
                        logging.info(f"✅ Found SD WebUI at {test_url}")
                        return test_url
                    else:
                        logging.debug(f"❌ {test_url} responded but not SD WebUI")
                except:
                    logging.debug(f"❌ {test_url} not accessible")
        
        logging.warning("❌ SD WebUI not found on any common locations")
        return None
    
    def check_connection(self) -> bool:
        """Check if Automatic1111 WebUI is running and accessible"""
        try:
            logging.info(f"Testing connection to SD WebUI at {self.base_url}")
            response = self.session.get(f"{self.base_url}/sdapi/v1/options", timeout=5)
            if response.status_code == 200:
                logging.info("✅ Stable Diffusion WebUI connected successfully")
                return True
            else:
                logging.warning(f"❌ SD WebUI responded with status: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            logging.warning(f"❌ Cannot connect to SD WebUI at {self.base_url} - Connection refused")
            logging.warning("💡 Make sure SD WebUI is running with: ./webui.sh --api --port 8001")
            return False
        except requests.exceptions.Timeout:
            logging.warning(f"❌ SD WebUI connection timeout at {self.base_url}")
            return False
        except requests.exceptions.RequestException as e:
            logging.warning(f"❌ SD WebUI connection error: {str(e)}")
            return False
    
    def get_models(self) -> list:
        """Get list of available models"""
        try:
            response = self.session.get(f"{self.base_url}/sdapi/v1/sd-models")
            if response.status_code == 200:
                models = response.json()
                logging.info(f"Found {len(models)} SD models available")
                return models
            else:
                logging.error(f"Failed to get models: {response.status_code}")
                return []
        except Exception as e:
            logging.error(f"Error getting models: {str(e)}")
            return []
    
    def generate_batch_images(self, prompt: str, output_dir: str, count: int = 6, **kwargs) -> List[str]:
        """Generate multiple images in batches for selection"""
        generated_paths = []
        
        # Generate in 2 batches of 3 for memory efficiency
        batch_size = 3
        num_batches = (count + batch_size - 1) // batch_size
        
        logging.info(f"🎨 Starting batch image generation: {count} images in {num_batches} batches")
        logging.info(f"📋 Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        
        for batch_num in range(num_batches):
            remaining = count - len(generated_paths)
            current_batch_size = min(batch_size, remaining)
            
            if current_batch_size <= 0:
                break
                
            logging.info(f"🔄 Batch {batch_num + 1}/{num_batches}: Generating {current_batch_size} images...")
            
            # Generate batch
            batch_paths = self._generate_batch(prompt, output_dir, current_batch_size, batch_num, **kwargs)
            
            if batch_paths:
                generated_paths.extend(batch_paths)
                logging.info(f"✅ Batch {batch_num + 1} completed: {len(batch_paths)} images generated")
                for i, path in enumerate(batch_paths):
                    logging.info(f"   📸 Image {len(generated_paths) - len(batch_paths) + i + 1}: {os.path.basename(path)}")
            else:
                logging.warning(f"❌ Batch {batch_num + 1} failed: No images generated")
        
        if generated_paths:
            logging.info(f"🎉 Batch generation complete: {len(generated_paths)}/{count} images successful")
        else:
            logging.error(f"💥 Batch generation failed: No images were generated")
        
        return generated_paths
    
    def _generate_batch(self, prompt: str, output_dir: str, batch_size: int, batch_num: int, **kwargs) -> List[str]:
        """Generate a single batch of images"""
        
        # Get negative prompt from kwargs or use default
        negative_prompt = kwargs.get("negative_prompt", "blurry, low quality, distorted, deformed, ugly, bad anatomy, bad hands, text, watermark, signature")
        
        # Enhance negative prompt with single person constraints
        enhanced_negative_prompt = f"{negative_prompt}, multiple people, crowd, group, two people, three people, many people, other person, additional person, background people, extra people, duplicate person"
        
        # Default parameters optimized for cyberrealistic model with ADetailer
        payload = {
            "prompt": prompt,
            "negative_prompt": enhanced_negative_prompt,
            "width": kwargs.get("width", 512),
            "height": kwargs.get("height", 768),
            "steps": kwargs.get("steps", 28),
            "cfg_scale": kwargs.get("cfg_scale", 7),
            "sampler_name": kwargs.get("sampler", "DPM++ 2M SDE"),
            "scheduler": kwargs.get("scheduler", "Karras"),
            "batch_size": batch_size,
            "n_iter": 1,
            "seed": kwargs.get("seed", -1),
            "restore_faces": kwargs.get("restore_faces", True),
            "tiling": False,
            "do_not_save_samples": True,
            "do_not_save_grid": True,
            
            # ADetailer configuration for face and hand fixing
            "alwayson_scripts": self._get_adetailer_scripts()
        }
        
        logging.info(f"⚙️  Batch {batch_num + 1} settings: {payload['width']}x{payload['height']}, {payload['steps']} steps, CFG={payload['cfg_scale']}")
        logging.info(f"🎲 Seed: {payload['seed']}, Sampler: {payload['sampler_name']}")
        logging.info(f"👤 Single person constraint enforced in negative prompt")
        
        # Log ADetailer status
        if self.available_adetailer_models:
            logging.info(f"🔧 ADetailer enabled with models: {', '.join(self.available_adetailer_models)}")
            face_models = [m for m in self.available_adetailer_models if 'face' in m.lower()]
            hand_models = [m for m in self.available_adetailer_models if 'hand' in m.lower()]
            person_models = [m for m in self.available_adetailer_models if 'person' in m.lower() or 'yolov8x' in m.lower()]
            
            enhancements = []
            if face_models: enhancements.append("Single person face fixing")
            if hand_models: enhancements.append("Single person hand fixing") 
            if person_models: enhancements.append("Single person refinement")
            
            if enhancements:
                logging.info(f"✨ Quality enhancements: {', '.join(enhancements)}")
        else:
            logging.info(f"ℹ️  ADetailer not available - using standard generation with single person constraints")
        
        try:
            logging.info(f"📡 Sending request to SD WebUI...")
            import time
            start_time = time.time()
            
            response = self.session.post(
                f"{self.base_url}/sdapi/v1/txt2img",
                json=payload,
                timeout=180  # Longer timeout for batches
            )
            
            generation_time = time.time() - start_time
            logging.info(f"⏱️  SD WebUI response received in {generation_time:.1f}s")
            
            if response.status_code == 200:
                result = response.json()
                
                if 'images' in result and len(result['images']) > 0:
                    batch_paths = []
                    logging.info(f"🖼️  Processing {len(result['images'])} generated images...")
                    
                    for i, image_b64 in enumerate(result['images']):
                        # Generate unique filename for each image in batch
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"batch_{batch_num}_{i}_{timestamp}.png"
                        output_path = os.path.join(output_dir, filename)
                        
                        logging.info(f"💾 Saving image {i + 1}/{len(result['images'])}: {filename}")
                        
                        # Decode and save image
                        image_data = base64.b64decode(image_b64)
                        with open(output_path, 'wb') as f:
                            f.write(image_data)
                        
                        file_size = len(image_data) / 1024  # KB
                        logging.info(f"   ✅ Saved: {filename} ({file_size:.1f} KB)")
                        
                        batch_paths.append(output_path)
                    
                    # Log generation info if available
                    if 'info' in result:
                        try:
                            info = json.loads(result['info'])
                            if 'seed' in info:
                                logging.info(f"🎲 Actual seed used: {info['seed']}")
                        except:
                            pass
                    
                    return batch_paths
                else:
                    logging.error(f"❌ No images in batch response from SD WebUI")
                    return []
            else:
                logging.error(f"❌ SD WebUI batch generation failed: HTTP {response.status_code}")
                logging.error(f"   Response: {response.text[:200]}...")
                return []
                
        except requests.exceptions.Timeout:
            logging.error(f"⏰ Batch generation timed out after 180 seconds")
            return []
        except Exception as e:
            logging.error(f"💥 Error in batch generation: {str(e)}")
            return []

    def generate_image(self, prompt: str, output_path: str, **kwargs) -> Optional[str]:
        """Generate image using Stable Diffusion with ADetailer"""
        
        # Get negative prompt from kwargs or use default
        negative_prompt = kwargs.get("negative_prompt", "blurry, low quality, distorted, deformed, ugly, bad anatomy, bad hands, text, watermark, signature")
        
        # Enhance negative prompt with single person constraints
        enhanced_negative_prompt = f"{negative_prompt}, multiple people, crowd, group, two people, three people, many people, other person, additional person, background people, extra people, duplicate person"
        
        # Default parameters optimized for cyberrealistic model with ADetailer
        payload = {
            "prompt": prompt,
            "negative_prompt": enhanced_negative_prompt,
            "width": kwargs.get("width", 512),
            "height": kwargs.get("height", 768),
            "steps": kwargs.get("steps", 28),
            "cfg_scale": kwargs.get("cfg_scale", 7),
            "sampler_name": kwargs.get("sampler", "DPM++ 2M SDE"),
            "scheduler": kwargs.get("scheduler", "Karras"),
            "batch_size": 1,
            "n_iter": 1,
            "seed": kwargs.get("seed", -1),
            "restore_faces": kwargs.get("restore_faces", True),
            "tiling": False,
            "do_not_save_samples": True,
            "do_not_save_grid": True,
            
            # ADetailer configuration for single image generation
            "alwayson_scripts": self._get_adetailer_scripts()
        }
        
        logging.info(f"Generating image with SD: {prompt[:100]}...")
        logging.info(f"Parameters: {payload['width']}x{payload['height']}, steps={payload['steps']}, cfg={payload['cfg_scale']}, seed={payload['seed']}")
        
        # Log ADetailer status for single image generation
        if self.available_adetailer_models:
            logging.info(f"🔧 ADetailer enabled with models: {', '.join(self.available_adetailer_models)}")
        else:
            logging.info(f"ℹ️  ADetailer not available - using standard generation")
        
        try:
            logging.info(f"Sending request to {self.base_url}/sdapi/v1/txt2img")
            response = self.session.post(
                f"{self.base_url}/sdapi/v1/txt2img",
                json=payload,
                timeout=120  # 2 minutes timeout for generation
            )
            logging.info(f"SD API response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                logging.info(f"SD API returned result with keys: {list(result.keys())}")
                
                if 'images' in result and len(result['images']) > 0:
                    logging.info(f"SD API returned {len(result['images'])} images")
                    # Decode base64 image
                    image_data = base64.b64decode(result['images'][0])
                    logging.info(f"Decoded image data size: {len(image_data)} bytes")
                    
                    # Save image
                    with open(output_path, 'wb') as f:
                        f.write(image_data)
                    
                    logging.info(f"✅ SD image generated successfully: {output_path}")
                    
                    # Log generation info if available
                    if 'info' in result:
                        info = json.loads(result['info'])
                        if 'seed' in info:
                            logging.info(f"Generated with actual seed: {info['seed']}")
                    
                    return output_path
                else:
                    logging.error(f"❌ No images returned from SD WebUI. Result: {result}")
                    return None
            else:
                logging.error(f"SD generation failed: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logging.error("SD generation timed out (2 minutes)")
            return None
        except Exception as e:
            logging.error(f"Error during SD generation: {str(e)}")
            return None
    
    def get_current_model(self) -> Optional[str]:
        """Get currently loaded model"""
        try:
            response = self.session.get(f"{self.base_url}/sdapi/v1/options")
            if response.status_code == 200:
                options = response.json()
                model = options.get('sd_model_checkpoint', 'Unknown')
                logging.info(f"Current SD model: {model}")
                return model
            return None
        except Exception as e:
            logging.error(f"Error getting current model: {str(e)}")
            return None
    
    def set_model(self, model_name: str) -> bool:
        """Set the active model"""
        try:
            payload = {"sd_model_checkpoint": model_name}
            response = self.session.post(
                f"{self.base_url}/sdapi/v1/options",
                json=payload
            )
            
            if response.status_code == 200:
                logging.info(f"Successfully switched to model: {model_name}")
                return True
            else:
                logging.error(f"Failed to switch model: {response.status_code}")
                return False
        except Exception as e:
            logging.error(f"Error switching model: {str(e)}")
            return False
    
    def _check_roop_availability(self):
        """Check if Roop extension is available and enabled"""
        try:
            response = self.session.get(f"{self.base_url}/sdapi/v1/extensions")
            if response.status_code == 200:
                extensions = response.json()
                for ext in extensions:
                    if ('roop' in ext.get('name', '').lower() or 
                        'roop' in ext.get('title', '').lower()) and ext.get('enabled', False):
                        self.roop_available = True
                        logging.info(f"✅ Roop extension found and enabled: {ext.get('name', 'Unknown')}")
                        return
                
                logging.warning("⚠️  Roop extension not found or not enabled")
            else:
                logging.warning(f"Failed to check extensions: {response.status_code}")
        except Exception as e:
            logging.error(f"Error checking Roop availability: {str(e)}")
            self.roop_available = False
    
    def _encode_image_to_base64(self, image_path: str) -> Optional[str]:
        """Encode image file to base64 string for API"""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save to base64
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                return img_base64
        except Exception as e:
            logging.error(f"Error encoding image {image_path}: {str(e)}")
            return None
    
    def generate_image_with_face_swap(self, prompt: str, face_image_path: str, output_path: str, 
                                    width: int = 512, height: int = 768, steps: int = 30, 
                                    cfg_scale: float = 7, sampler: str = "DPM++ 2M SDE", 
                                    scheduler: str = "Karras", seed: int = -1, negative_prompt: str = "") -> bool:
        """Generate image with Roop face swapping"""
        
        if not self.roop_available:
            logging.warning("Roop extension not available - falling back to regular generation")
            return self._fallback_to_regular_generation(prompt, output_path, width, height, steps, cfg_scale, sampler, scheduler, seed, negative_prompt)
        
        # Encode face image
        face_b64 = self._encode_image_to_base64(face_image_path)
        if not face_b64:
            logging.error(f"Failed to encode face image: {face_image_path} - falling back to regular generation")
            return self._fallback_to_regular_generation(prompt, output_path, width, height, steps, cfg_scale, sampler, scheduler, seed, negative_prompt)
        
        logging.info(f"🎭 Generating image with Roop face swap...")
        logging.info(f"Face source: {os.path.basename(face_image_path)}")
        logging.info(f"Parameters: {width}x{height}, steps={steps}, cfg={cfg_scale}, seed={seed}")
        
        # Enhanced negative prompt for single person constraint
        enhanced_negative_prompt = f"{negative_prompt}, multiple people, crowd, group, two people, three people, many people, other person, additional person, background people, extra people, duplicate person"
        
        # Simplified Roop configuration to avoid face_swapper initialization issues
        payload = {
            "prompt": prompt,
            "negative_prompt": enhanced_negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "sampler_name": sampler,
            "scheduler": scheduler,
            "seed": seed,
            "batch_size": 1,
            "n_iter": 1,
            "restore_faces": False,  # Let Roop handle face restoration
            "tiling": False,
            "do_not_save_samples": True,
            "do_not_save_grid": True,
            
            # Simplified Roop extension parameters
            "alwayson_scripts": {
                "roop": {
                    "args": [
                        face_b64,           # Source face image (base64)
                        True,               # Enable Roop
                        "0",                # Face index (first face detected)
                        face_b64,           # Reference image (same as source for consistency)
                        True,               # Restore face after swap
                        0.7,                # Face restoration visibility (lower value for stability)
                        0.8,                # Codeformer weight (if available)
                        False,              # Restore face first (avoid conflicts)
                        "CodeFormer",       # Upscaler method
                        1,                  # Upscaler scale
                        0.8,                # Upscaler visibility
                        False,              # Swap in source image
                        True,               # Swap in generated image
                        0,                  # Console logging level (reduced)
                        0,                  # Gender detection source
                        0,                  # Gender detection target
                        False,              # Save original before swap
                        0.5,                # Face mask correction (lower for stability)
                        0,                  # Face mask blur
                        False,              # Face mask padding
                        "max"               # Face parsing model
                    ]
                }
            }
        }
        
        # Only add ADetailer for hand fixing (avoid face conflicts with Roop)
        if self.available_adetailer_models:
            hand_only_scripts = self._get_hand_only_adetailer_scripts()
            if hand_only_scripts:
                payload["alwayson_scripts"].update(hand_only_scripts)
                logging.info(f"🔧 Hand-only ADetailer + Roop combination enabled")
        
        try:
            logging.info(f"📡 Sending Roop face swap request to {self.base_url}/sdapi/v1/txt2img")
            
            response = self.session.post(
                f"{self.base_url}/sdapi/v1/txt2img",
                json=payload,
                timeout=120
            )
            
            logging.info(f"📡 SD API response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                logging.info(f"📊 SD API returned result with keys: {list(result.keys())}")
                
                if 'images' in result and len(result['images']) > 0:
                    logging.info(f"🖼️  SD API returned {len(result['images'])} images")
                    
                    # Decode and save the first image
                    image_b64 = result['images'][0]
                    image_data = base64.b64decode(image_b64)
                    
                    with open(output_path, 'wb') as f:
                        f.write(image_data)
                    
                    logging.info(f"✅ Roop face-swapped image generated successfully: {output_path}")
                    
                    # Log generation info if available
                    if 'info' in result:
                        try:
                            info = json.loads(result['info'])
                            actual_seed = info.get('seed', 'Unknown')
                            logging.info(f"🎲 Generated with actual seed: {actual_seed}")
                        except:
                            pass
                    
                    return True
                else:
                    logging.error("❌ No images returned from SD API - falling back to regular generation")
                    return self._fallback_to_regular_generation(prompt, output_path, width, height, steps, cfg_scale, sampler, scheduler, seed, negative_prompt)
            else:
                logging.error(f"❌ SD API error: {response.status_code} - falling back to regular generation")
                logging.debug(f"Response text: {response.text[:500]}")
                return self._fallback_to_regular_generation(prompt, output_path, width, height, steps, cfg_scale, sampler, scheduler, seed, negative_prompt)
                
        except Exception as e:
            logging.error(f"💥 Error in Roop face swap generation: {str(e)} - falling back to regular generation")
            return self._fallback_to_regular_generation(prompt, output_path, width, height, steps, cfg_scale, sampler, scheduler, seed, negative_prompt)
    
    def _fallback_to_regular_generation(self, prompt: str, output_path: str, width: int, height: int, 
                                       steps: int, cfg_scale: float, sampler: str, scheduler: str, 
                                       seed: int, negative_prompt: str) -> bool:
        """Fallback to regular generation when Roop fails"""
        logging.info("🔄 Falling back to regular image generation with character consistency")
        
        result = self.generate_image(
            prompt=prompt,
            output_path=output_path,
            width=width,
            height=height,
            steps=steps,
            cfg_scale=cfg_scale,
            sampler=sampler,
            scheduler=scheduler,
            seed=seed,
            negative_prompt=negative_prompt
        )
        
        return result is not None
    
    def _get_hand_only_adetailer_scripts(self) -> Dict:
        """Get ADetailer configuration with only hand fixing to avoid face conflicts with Roop"""
        if not self.available_adetailer_models:
            return {}
        
        configs = []
        
        # Only add hand detection to avoid conflicts with Roop face swapping
        if "hand_yolov8n.pt" in self.available_adetailer_models:
            configs.append({
                "ad_model": "hand_yolov8n.pt",
                "ad_prompt": "single person hands, perfect hands, detailed fingers, natural pose, correct anatomy, one person only",
                "ad_negative_prompt": "deformed hands, bad fingers, extra fingers, missing fingers, blurry hands, multiple people, other person hands",
                "ad_confidence": 0.3,
                "ad_mask_blur": 4,
                "ad_denoising_strength": 0.5,
                "ad_inpaint_only_masked": True,
                "ad_inpaint_only_masked_padding": 32,
                "ad_steps": 20,
                "ad_cfg_scale": 7.0,
                "ad_sampler": "DPM++ 2M SDE"
            })
        
        if not configs:
            return {}
        
        # Build the script args
        script_args = [
            True,   # Enable ADetailer
            False,  # Skip img2img
        ]
        script_args.extend(configs)
        
        logging.debug(f"🔧 Hand-only ADetailer configured for Roop compatibility")
        
        return {
            "ADetailer": {
                "args": script_args
            }
        }
    
    def generate_batch_with_face_swap(self, prompt: str, face_image_path: str, output_dir: str,
                                    count: int = 6, width: int = 512, height: int = 768, 
                                    steps: int = 30, cfg_scale: float = 7, sampler: str = "DPM++ 2M SDE",
                                    scheduler: str = "Karras", seed: int = -1, negative_prompt: str = "") -> List[str]:
        """Generate batch of images with consistent face swapping"""
        
        logging.info(f"🎭 Starting batch generation with Roop face swap")
        logging.info(f"Face source: {os.path.basename(face_image_path)}")
        logging.info(f"Generating {count} images with face consistency")
        
        # Check if Roop is available before starting batch
        if not self.roop_available:
            logging.warning("🔄 Roop not available - falling back to regular batch generation with character consistency")
            return self.generate_batch_images(
                prompt=prompt,
                output_dir=output_dir,
                count=count,
                width=width,
                height=height,
                steps=steps,
                cfg_scale=cfg_scale,
                sampler=sampler,
                scheduler=scheduler,
                seed=seed,
                negative_prompt=negative_prompt
            )
        
        batch_paths = []
        failed_attempts = 0
        max_failures = 3  # Allow some failures before falling back completely
        
        # Generate images in 2 batches of 3 (as per existing system)
        batch_size = 3
        num_batches = (count + batch_size - 1) // batch_size
        
        for batch_num in range(num_batches):
            remaining = count - len(batch_paths)
            current_batch_size = min(batch_size, remaining)
            
            logging.info(f"🔄 Generating batch {batch_num + 1}/{num_batches} with {current_batch_size} images")
            
            for i in range(current_batch_size):
                # Use different seeds for variation
                batch_seed = seed + (batch_num * batch_size + i) if seed != -1 else -1
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"roop_batch_{batch_num}_{i}_{timestamp}.png"
                output_path = os.path.join(output_dir, filename)
                
                success = self.generate_image_with_face_swap(
                    prompt=prompt,
                    face_image_path=face_image_path,
                    output_path=output_path,
                    width=width,
                    height=height,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    sampler=sampler,
                    scheduler=scheduler,
                    seed=batch_seed,
                    negative_prompt=negative_prompt
                )
                
                if success:
                    batch_paths.append(output_path)
                    logging.info(f"✅ Generated image {len(batch_paths)}/{count}: {filename}")
                    failed_attempts = 0  # Reset failure counter on success
                else:
                    failed_attempts += 1
                    logging.error(f"❌ Failed to generate image {batch_num}_{i} (failure #{failed_attempts})")
                    
                    # If too many failures, fall back to regular generation for remaining images
                    if failed_attempts >= max_failures:
                        logging.warning(f"🔄 Too many Roop failures ({failed_attempts}), falling back to regular generation for remaining images")
                        remaining_count = count - len(batch_paths)
                        if remaining_count > 0:
                            fallback_paths = self.generate_batch_images(
                                prompt=prompt,
                                output_dir=output_dir,
                                count=remaining_count,
                                width=width,
                                height=height,
                                steps=steps,
                                cfg_scale=cfg_scale,
                                sampler=sampler,
                                scheduler=scheduler,
                                seed=seed + len(batch_paths),
                                negative_prompt=negative_prompt
                            )
                            batch_paths.extend(fallback_paths)
                        break
            
            # Break outer loop if we hit max failures
            if failed_attempts >= max_failures:
                break
        
        logging.info(f"🎭 Batch generation complete: {len(batch_paths)}/{count} images generated with face swap")
        return batch_paths
    
    def get_progress(self) -> Dict:
        """Get current generation progress"""
        try:
            response = self.session.get(f"{self.base_url}/sdapi/v1/progress")
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            logging.error(f"Error getting progress: {str(e)}")
            return {}
    
    def interrupt_generation(self) -> bool:
        """Interrupt current generation"""
        try:
            response = self.session.post(f"{self.base_url}/sdapi/v1/interrupt")
            return response.status_code == 200
        except Exception as e:
            logging.error(f"Error interrupting generation: {str(e)}")
            return False
            return False