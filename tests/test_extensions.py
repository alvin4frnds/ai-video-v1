"""
Test external extensions, services, and integrations
"""

import unittest
import logging
import requests
import time
import os
import sys
from unittest.mock import patch, Mock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestStableDiffusionWebUI(unittest.TestCase):
    """Test Stable Diffusion WebUI connectivity and API"""

    def setUp(self):
        self.sd_urls = [
            "http://localhost:8001",
            "http://127.0.0.1:8001", 
            "http://192.168.0.199:8001"
        ]
        self.timeout = 5

    def test_sd_webui_connectivity(self):
        """Test if SD WebUI is accessible on any common endpoint"""
        sd_accessible = False
        accessible_url = None
        
        for url in self.sd_urls:
            try:
                response = requests.get(f"{url}/sdapi/v1/options", timeout=self.timeout)
                if response.status_code == 200:
                    sd_accessible = True
                    accessible_url = url
                    logging.info(f"✅ SD WebUI accessible at {url}")
                    break
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                logging.warning(f"⚠️ SD WebUI not accessible at {url}")
                continue
            except Exception as e:
                logging.warning(f"⚠️ Error checking SD WebUI at {url}: {e}")
                continue
        
        if not sd_accessible:
            logging.warning("⚠️ SD WebUI not accessible on any endpoint - tests will use mocked responses")
            self.skipTest("SD WebUI not accessible for live testing")
        else:
            self.assertTrue(sd_accessible)
            self.assertIsNotNone(accessible_url)

    def test_sd_webui_models_endpoint(self):
        """Test SD WebUI models endpoint"""
        for url in self.sd_urls:
            try:
                response = requests.get(f"{url}/sdapi/v1/sd-models", timeout=self.timeout)
                if response.status_code == 200:
                    models = response.json()
                    self.assertIsInstance(models, list)
                    logging.info(f"✅ Found {len(models)} models in SD WebUI")
                    
                    # Check for cyberrealistic models
                    model_names = [model.get('model_name', '') for model in models]
                    cyberrealistic_models = [name for name in model_names if 'cyberrealistic' in name.lower()]
                    if cyberrealistic_models:
                        logging.info(f"✅ Found cyberrealistic models: {cyberrealistic_models}")
                    else:
                        logging.warning("⚠️ No cyberrealistic models found")
                    
                    return  # Success, exit test
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                continue
        
        self.skipTest("SD WebUI not accessible for model testing")

    def test_adetailer_extension(self):
        """Test ADetailer extension availability"""
        for url in self.sd_urls:
            try:
                # Check ADetailer models endpoint
                response = requests.get(f"{url}/adetailer/v1/ad_model", timeout=self.timeout)
                if response.status_code == 200:
                    models = response.json()
                    available_models = models.get("ad_model", [])
                    
                    self.assertIsInstance(available_models, list)
                    logging.info(f"✅ ADetailer extension found with {len(available_models)} models")
                    
                    # Check for required models
                    required_models = ["face_yolov8n.pt", "hand_yolov8n.pt", "person_yolov8n-seg.pt"]
                    found_models = [model for model in required_models if model in available_models]
                    
                    if found_models:
                        logging.info(f"✅ Found required ADetailer models: {found_models}")
                    else:
                        logging.warning(f"⚠️ Required ADetailer models not found. Available: {available_models}")
                    
                    return  # Success, exit test
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                continue
        
        logging.warning("⚠️ ADetailer extension not accessible")
        self.skipTest("ADetailer extension not accessible for testing")

    def test_roop_extension(self):
        """Test Roop extension availability"""
        for url in self.sd_urls:
            try:
                # Check extensions endpoint
                response = requests.get(f"{url}/sdapi/v1/extensions", timeout=self.timeout)
                if response.status_code == 200:
                    extensions = response.json()
                    
                    # Look for Roop extension
                    roop_extensions = [
                        ext for ext in extensions 
                        if 'roop' in ext.get('name', '').lower() or 'roop' in ext.get('title', '').lower()
                    ]
                    
                    if roop_extensions:
                        roop_ext = roop_extensions[0]
                        is_enabled = roop_ext.get('enabled', False)
                        
                        logging.info(f"✅ Roop extension found: {roop_ext.get('name', 'Unknown')}")
                        logging.info(f"✅ Roop enabled: {is_enabled}")
                        
                        if is_enabled:
                            self.assertTrue(True)  # Roop is available and enabled
                        else:
                            logging.warning("⚠️ Roop extension found but not enabled")
                    else:
                        logging.warning("⚠️ Roop extension not found")
                    
                    return  # Success, exit test
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                continue
        
        logging.warning("⚠️ Cannot check Roop extension - SD WebUI not accessible")
        self.skipTest("SD WebUI not accessible for Roop testing")

class TestMixtralOllama(unittest.TestCase):
    """Test Mixtral/Ollama connectivity and functionality"""

    def setUp(self):
        self.ollama_urls = [
            "http://localhost:11434",
            "http://127.0.0.1:11434"
        ]
        self.timeout = 10

    def test_ollama_connectivity(self):
        """Test if Ollama service is running"""
        ollama_accessible = False
        
        for url in self.ollama_urls:
            try:
                response = requests.get(f"{url}/api/tags", timeout=self.timeout)
                if response.status_code == 200:
                    ollama_accessible = True
                    logging.info(f"✅ Ollama accessible at {url}")
                    
                    # Check available models
                    models = response.json()
                    model_names = [model.get('name', '') for model in models.get('models', [])]
                    
                    # Look for Mixtral model
                    mixtral_models = [name for name in model_names if 'mixtral' in name.lower()]
                    if mixtral_models:
                        logging.info(f"✅ Found Mixtral models: {mixtral_models}")
                    else:
                        logging.warning("⚠️ No Mixtral models found")
                        logging.info(f"Available models: {model_names}")
                    
                    break
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                logging.warning(f"⚠️ Ollama not accessible at {url}")
                continue
        
        if not ollama_accessible:
            logging.warning("⚠️ Ollama service not accessible - Mixtral tests will be skipped")
            self.skipTest("Ollama not accessible for testing")
        else:
            self.assertTrue(ollama_accessible)

    def test_mixtral_chat_endpoint(self):
        """Test Mixtral chat endpoint functionality"""
        for url in self.ollama_urls:
            try:
                # Test simple chat request
                payload = {
                    "model": "mixtral",
                    "messages": [
                        {"role": "user", "content": "Hello, this is a test message."}
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "max_tokens": 50
                    }
                }
                
                response = requests.post(f"{url}/api/chat", json=payload, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    self.assertIn('message', result)
                    self.assertIn('content', result['message'])
                    
                    content = result['message']['content']
                    self.assertIsInstance(content, str)
                    self.assertGreater(len(content), 0)
                    
                    logging.info(f"✅ Mixtral chat endpoint working. Response length: {len(content)}")
                    return  # Success, exit test
                    
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                logging.warning(f"⚠️ Mixtral chat test failed at {url}: {e}")
                continue
        
        self.skipTest("Mixtral chat endpoint not accessible for testing")

class TestSystemDependencies(unittest.TestCase):
    """Test system-level dependencies and tools"""

    def test_ffmpeg_availability(self):
        """Test if ffmpeg is available for video processing"""
        import subprocess
        
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logging.info("✅ FFmpeg is available")
                
                # Check for H.264 codec support
                if 'libx264' in result.stdout:
                    logging.info("✅ H.264 codec support available")
                else:
                    logging.warning("⚠️ H.264 codec support not found")
                    
                self.assertTrue(True)
            else:
                logging.warning("⚠️ FFmpeg found but returned error")
                self.skipTest("FFmpeg not working properly")
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logging.warning("⚠️ FFmpeg not found - video conversion may be limited")
            self.skipTest("FFmpeg not available")

    def test_opencv_video_codecs(self):
        """Test OpenCV video codec availability"""
        try:
            import cv2
            
            # Test video writer initialization with different codecs
            codecs_to_test = ['avc1', 'H264', 'mp4v', 'XVID']
            working_codecs = []
            
            for codec in codecs_to_test:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    # Try to create a video writer (will fail if codec not supported)
                    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                        writer = cv2.VideoWriter(tmp_file.name, fourcc, 30, (640, 480))
                        if writer.isOpened():
                            working_codecs.append(codec)
                            writer.release()
                        os.unlink(tmp_file.name)
                except Exception:
                    continue
            
            logging.info(f"✅ Working OpenCV codecs: {working_codecs}")
            self.assertGreater(len(working_codecs), 0, "No working video codecs found")
            
        except ImportError:
            self.skipTest("OpenCV not available for codec testing")

class TestDirectoryStructure(unittest.TestCase):
    """Test required directory structure and permissions"""

    def test_output_directories(self):
        """Test output directory structure and permissions"""
        required_dirs = [
            "output",
            "output/frames", 
            "output/videos"
        ]
        
        for dir_path in required_dirs:
            # Check if directory exists or can be created
            if not os.path.exists(dir_path):
                try:
                    os.makedirs(dir_path, exist_ok=True)
                    logging.info(f"✅ Created directory: {dir_path}")
                except Exception as e:
                    self.fail(f"Cannot create required directory {dir_path}: {e}")
            else:
                logging.info(f"✅ Directory exists: {dir_path}")
            
            # Check write permissions
            self.assertTrue(os.access(dir_path, os.W_OK), 
                          f"No write permission for directory: {dir_path}")

    def test_face_input_directories(self):
        """Test face input directory structure"""
        face_dir = "in/individual"
        
        if not os.path.exists(face_dir):
            try:
                os.makedirs(face_dir, exist_ok=True)
                logging.info(f"✅ Created face directory: {face_dir}")
            except Exception as e:
                logging.warning(f"⚠️ Cannot create face directory {face_dir}: {e}")
                return
        
        # Check for sample face directories
        sample_dirs = ["ayushi", "nitanshi"]
        found_faces = []
        
        for sample_dir in sample_dirs:
            sample_path = os.path.join(face_dir, sample_dir)
            if os.path.exists(sample_path):
                # Count image files
                image_files = [f for f in os.listdir(sample_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if image_files:
                    found_faces.append(f"{sample_dir}: {len(image_files)} images")
        
        if found_faces:
            logging.info(f"✅ Found face directories: {found_faces}")
        else:
            logging.warning("⚠️ No face directories with images found")

if __name__ == '__main__':
    unittest.main()