import requests
import json
import logging
import base64
from io import BytesIO
from PIL import Image
import os
from typing import Dict, Optional

class StableDiffusionClient:
    """Client for interacting with Automatic1111 Stable Diffusion WebUI"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.session = requests.Session()
        logging.info(f"Initialized Stable Diffusion client with base URL: {base_url}")
    
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
    
    def generate_image(self, prompt: str, output_path: str, **kwargs) -> Optional[str]:
        """Generate image using Stable Diffusion"""
        
        # Default parameters optimized for cyberrealistic model
        payload = {
            "prompt": prompt,
            "negative_prompt": "blurry, low quality, distorted, deformed, ugly, bad anatomy, bad hands, text, watermark, signature",
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
            "do_not_save_grid": True
        }
        
        logging.info(f"Generating image with SD: {prompt[:100]}...")
        logging.info(f"Parameters: {payload['width']}x{payload['height']}, steps={payload['steps']}, cfg={payload['cfg_scale']}, seed={payload['seed']}")
        
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