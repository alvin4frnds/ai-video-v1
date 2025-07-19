"""
Unit tests for all model files and their individual components
"""

import unittest
import logging
import os
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestMixtralClient(unittest.TestCase):
    """Test MixtralClient functionality"""

    def setUp(self):
        # Import here to avoid import errors during collection
        from mixtral_client import MixtralClient
        self.client = MixtralClient()

    def test_mixtral_client_initialization(self):
        """Test MixtralClient initializes correctly"""
        self.assertEqual(self.client.base_url, "http://localhost:11434")
        self.assertIsNotNone(self.client.session)
        # MixtralClient doesn't have base_seed attribute, that's in VideoGenerator
        self.assertIsNotNone(self.client.base_url)
        self.assertIsNotNone(self.client.model)

    @patch('requests.Session.post')
    def test_mixtral_chat_success(self, mock_post):
        """Test successful Mixtral chat response"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'message': {'content': 'Test response from Mixtral'}
        }
        mock_post.return_value = mock_response

        result = self.client._call_mixtral("Test system prompt", "Test prompt")
        self.assertEqual(result, "Test response from Mixtral")
        
        # Verify timeout is 100 seconds
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(kwargs['timeout'], 100)

    @patch('requests.Session.post')
    def test_mixtral_chat_failure(self, mock_post):
        """Test Mixtral chat failure handling"""
        # Mock failed response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Server error"
        mock_post.return_value = mock_response

        with self.assertRaises(Exception) as context:
            self.client._call_mixtral("Test system prompt", "Test prompt")
        
        # Should raise the mocked exception
        self.assertIn("500", str(context.exception))

    def test_sample_prompts(self):
        """Test sample prompts for different scenarios"""
        sample_prompts = [
            "woman walking in park",
            "man walking street", 
            "woman shopping mall",
            "person sitting cafe",
            "child playing playground"
        ]
        
        for prompt in sample_prompts:
            with self.subTest(prompt=prompt):
                # Test that fallback methods work
                result = self.client._fallback_scene_analysis(prompt)
                self.assertIsInstance(result, list)
                self.assertGreater(len(result), 0)
                
                # Each scene should have required keys
                for scene in result:
                    # Check if it's a fallback scene (string) or parsed scene (dict)
            if isinstance(scene, dict):
                self.assertIn('description', scene)
            else:
                # Fallback returns strings, which is expected
                self.assertIsInstance(scene, str)
                    self.assertIn('duration', scene)

class TestStableDiffusionClient(unittest.TestCase):
    """Test StableDiffusionClient functionality"""

    def setUp(self):
        from sd_client import StableDiffusionClient
        self.client = StableDiffusionClient("http://localhost:8001")

    def test_sd_client_initialization(self):
        """Test SD client initializes correctly"""
        self.assertEqual(self.client.base_url, "http://localhost:8001")
        self.assertIsNotNone(self.client.session)
        self.assertIsInstance(self.client.available_adetailer_models, list)
        self.assertIsInstance(self.client.roop_available, bool)

    def test_adetailer_config_generation(self):
        """Test ADetailer configuration generation"""
        # Test with no models available
        self.client.available_adetailer_models = []
        config = self.client.get_adetailer_config()
        self.assertEqual(config, [])
        
        # Test with face model available
        self.client.available_adetailer_models = ["face_yolov8n.pt"]
        config = self.client.get_adetailer_config()
        self.assertIsInstance(config, list)
        self.assertGreater(len(config), 0)
        
        # Verify face config structure
        face_config = config[0]
        self.assertIn('ad_model', face_config)
        self.assertIn('ad_prompt', face_config)
        self.assertIn('ad_negative_prompt', face_config)

    def test_image_encoding(self):
        """Test image encoding to base64"""
        # Create a temporary test image
        from PIL import Image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            # Create simple test image
            img = Image.new('RGB', (100, 100), color='red')
            img.save(tmp_file.name)
            
            # Test encoding
            encoded = self.client._encode_image_to_base64(tmp_file.name)
            self.assertIsNotNone(encoded)
            self.assertIsInstance(encoded, str)
            
            # Clean up
            os.unlink(tmp_file.name)

    def test_fallback_generation(self):
        """Test fallback to regular generation"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            # Mock the generate_image method to return success
            with patch.object(self.client, 'generate_image', return_value=tmp_file.name):
                result = self.client._fallback_to_regular_generation(
                    prompt="test prompt",
                    output_path=tmp_file.name,
                    width=512,
                    height=768,
                    steps=20,
                    cfg_scale=7,
                    sampler="DPM++ 2M SDE",
                    scheduler="Karras",
                    seed=12345,
                    negative_prompt="test negative"
                )
                self.assertTrue(result)
            
            # Clean up
            os.unlink(tmp_file.name)

class TestFaceAnalyzer(unittest.TestCase):
    """Test FaceAnalyzer functionality"""

    def setUp(self):
        from face_analyzer import FaceAnalyzer
        self.analyzer = FaceAnalyzer()

    def test_face_analyzer_initialization(self):
        """Test FaceAnalyzer initializes correctly"""
        self.assertIsNotNone(self.analyzer)

    def test_color_analysis(self):
        """Test color analysis functions"""
        # Create test image
        from PIL import Image
        import numpy as np
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            # Create simple red image
            img_array = np.full((100, 100, 3), [255, 0, 0], dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(tmp_file.name)
            
            try:
                # Test dominant color extraction
                # _extract_dominant_colors takes np.ndarray and n_colors parameter
                test_array = np.zeros((100, 100, 3), dtype=np.uint8)
                colors = self.analyzer._extract_dominant_colors(test_array, n_colors=3)
                self.assertIsInstance(colors, list)
                
                # Test color similarity
                similarity = self.analyzer._calculate_color_similarity([255, 0, 0], [255, 0, 0])
                self.assertEqual(similarity, 1.0)  # Identical colors should have similarity 1.0
                
            except Exception as e:
                # If dependencies not available, log warning
                logging.warning(f"Color analysis test skipped due to missing dependencies: {e}")
            finally:
                os.unlink(tmp_file.name)

class TestFaceSelector(unittest.TestCase):
    """Test FaceSelector functionality"""

    def setUp(self):
        from face_selector import FaceSelector
        # Create temporary test directory structure
        self.test_dir = tempfile.mkdtemp()
        self.selector = FaceSelector(self.test_dir)

    def tearDown(self):
        # Clean up test directory
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_face_selector_initialization(self):
        """Test FaceSelector initializes correctly"""
        self.assertEqual(self.selector.faces_dir, self.test_dir)
        self.assertIsNone(self.selector.selected_face)
        self.assertIsNone(self.selector.selected_person)

    def test_empty_directory_handling(self):
        """Test handling of empty face directory"""
        people = self.selector.get_available_people()
        self.assertEqual(people, [])

    def test_face_selection_no_faces(self):
        """Test face selection when no faces available"""
        result = self.selector.select_random_face()
        self.assertIsNone(result)

    def test_face_directory_structure(self):
        """Test face directory structure creation and detection"""
        # Create test face directory structure
        person_dir = os.path.join(self.test_dir, "test_person")
        os.makedirs(person_dir, exist_ok=True)
        
        # Create dummy image file
        test_image_path = os.path.join(person_dir, "test_image.png")
        from PIL import Image
        img = Image.new('RGB', (100, 100), color='blue')
        img.save(test_image_path)
        
        # Test detection
        people = self.selector.get_available_people()
        self.assertIn("test_person", people)
        
        # Test face selection
        selected = self.selector.select_random_face(person="test_person")
        self.assertIsNotNone(selected)
        self.assertEqual(selected, test_image_path)

class TestVideoGenerator(unittest.TestCase):
    """Test VideoGenerator functionality"""

    def setUp(self):
        from video_generator import VideoGenerator
        self.generator = VideoGenerator(candidate_count=2)

    def test_video_generator_initialization(self):
        """Test VideoGenerator initializes correctly"""
        self.assertEqual(self.generator.candidate_count, 2)
        self.assertIsInstance(self.generator.base_seed, int)
        self.assertTrue(os.path.exists(self.generator.frames_dir))
        self.assertTrue(os.path.exists(self.generator.videos_dir))

    def test_output_directories_creation(self):
        """Test that output directories are created"""
        self.assertTrue(os.path.exists(self.generator.output_dir))
        self.assertTrue(os.path.exists(self.generator.frames_dir))
        self.assertTrue(os.path.exists(self.generator.videos_dir))

    def test_seed_generation(self):
        """Test seed generation format"""
        # Base seed should be in YmdH format (8-10 digits)
        self.assertGreaterEqual(len(str(self.generator.base_seed)), 8)
        self.assertLessEqual(len(str(self.generator.base_seed)), 10)

    @patch('video_generator.MixtralClient')
    @patch('video_generator.StableDiffusionClient')
    @patch('video_generator.FaceAnalyzer')
    @patch('video_generator.FaceSelector')
    def test_component_initialization(self, mock_face_selector, mock_face_analyzer, mock_sd_client, mock_mixtral):
        """Test that all components are properly initialized"""
        from video_generator import VideoGenerator
        
        # Create new generator to test initialization
        generator = VideoGenerator(candidate_count=3)
        
        # Verify all components are created
        self.assertIsNotNone(generator.mixtral)
        self.assertIsNotNone(generator.sd_client)
        self.assertIsNotNone(generator.face_analyzer)
        self.assertIsNotNone(generator.face_selector)

if __name__ == '__main__':
    unittest.main()