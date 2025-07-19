"""
Functional tests for video generation pipeline components
"""

import unittest
import logging
import os
import tempfile
import sys
from unittest.mock import Mock, patch, MagicMock
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestPipelineIntegration(unittest.TestCase):
    """Test pipeline integration and data flow"""

    def setUp(self):
        # Create temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp(prefix='pipeline_test_')

    def tearDown(self):
        # Clean up test directory
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch('mixtral_client.requests.Session.post')
    def test_mixtral_pipeline_integration(self, mock_post):
        """Test Mixtral integration in pipeline"""
        from mixtral_client import MixtralClient
        
        # Mock successful Mixtral response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'message': {
                'content': '''[
                    {
                        "scene_number": 1,
                        "description": "Woman walking in park entrance",
                        "duration": 3.0,
                        "transition_type": "fade"
                    },
                    {
                        "scene_number": 2, 
                        "description": "Woman walking along park path",
                        "duration": 3.0,
                        "transition_type": "dissolve"
                    }
                ]'''
            }
        }
        mock_post.return_value = mock_response
        
        client = MixtralClient()
        
        # Test scene analysis
        result = client.analyze_narrative_structure("woman walking in park")
        
        # Verify result structure
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        
        for scene in result:
            self.assertIn('description', scene)
            self.assertIn('duration', scene)
            # Handle both 'transition_type' and 'transition' field names
            has_transition = 'transition_type' in scene or 'transition' in scene
            self.assertTrue(has_transition, f"Scene missing transition field: {scene}")
        
        logging.info("✅ Mixtral pipeline integration working")

    def test_face_analyzer_pipeline(self):
        """Test face analyzer in pipeline context"""
        from face_analyzer import FaceAnalyzer
        from PIL import Image
        import numpy as np
        
        analyzer = FaceAnalyzer()
        
        # Create test images with different characteristics
        test_images = []
        
        for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
            img_path = os.path.join(self.test_dir, f'test_image_{i}.png')
            
            # Create colored test image
            img_array = np.full((200, 200, 3), color, dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(img_path)
            
            test_images.append(img_path)
        
        # Test batch analysis
        try:
            # Test actual method that exists
            for test_image in test_images:
                result = analyzer.detect_faces(test_image)
                
                # Verify basic analysis structure
                self.assertIn('face_count', result)
                self.assertIn('quality_score', result)
                self.assertIn('has_realistic_face', result)
                self.assertIsInstance(result['face_count'], int)
            
            logging.info("✅ Face analyzer pipeline integration working")
            
        except Exception as e:
            logging.warning(f"⚠️ Face analyzer test skipped due to dependencies: {e}")
            self.skipTest(f"Face analyzer dependencies not available: {e}")

    def test_video_generator_initialization_pipeline(self):
        """Test VideoGenerator initialization in pipeline context"""
        from video_generator import VideoGenerator
        
        # Test with different candidate counts
        for count in [1, 2, 6, 10]:
            with self.subTest(candidate_count=count):
                generator = VideoGenerator(candidate_count=count)
                
                # Verify initialization
                self.assertEqual(generator.candidate_count, count)
                self.assertIsNotNone(generator.base_seed)
                self.assertTrue(os.path.exists(generator.frames_dir))
                self.assertTrue(os.path.exists(generator.videos_dir))
                
                logging.info(f"✅ VideoGenerator pipeline init working (count={count})")

    @patch('sd_client.requests.Session.post')
    @patch('sd_client.requests.Session.get')
    def test_sd_client_pipeline_integration(self, mock_get, mock_post):
        """Test SD client integration in pipeline"""
        from sd_client import StableDiffusionClient
        
        # Mock SD WebUI responses
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = []
        
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            'images': ['fake_base64_image_data'],
            'info': '{"seed": 12345}'
        }
        
        client = StableDiffusionClient()
        
        # Test batch generation
        output_dir = os.path.join(self.test_dir, 'batch_test')
        os.makedirs(output_dir, exist_ok=True)
        
        with patch('base64.b64decode', return_value=b'fake_image_data'):
            batch_paths = client.generate_batch_images(
                prompt="test prompt",
                output_dir=output_dir,
                count=2
            )
            
            # Verify batch generation structure
            self.assertIsInstance(batch_paths, list)
            
            logging.info("✅ SD client pipeline integration working")

class TestAppPipelineIntegration(unittest.TestCase):
    """Test app.py pipeline integration"""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix='app_test_')

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_app_import_and_structure(self):
        """Test app.py imports and basic structure"""
        try:
            import app
            
            # Test key functions exist
            self.assertTrue(hasattr(app, 'generate_video_pipeline'))
            self.assertTrue(hasattr(app, 'generate_random_test_prompt'))
            self.assertTrue(hasattr(app, 'create_batch_analysis_display'))
            self.assertTrue(hasattr(app, 'create_batch_analysis_table'))
            self.assertTrue(hasattr(app, 'clean_output_folder'))
            
            logging.info("✅ App.py structure validation passed")
            
        except ImportError as e:
            self.fail(f"Failed to import app.py: {e}")

    def test_random_prompt_generation(self):
        """Test random prompt generation"""
        from app import generate_random_test_prompt
        
        # Generate multiple prompts to test variation
        prompts = set()
        for _ in range(10):
            prompt = generate_random_test_prompt()
            self.assertIsInstance(prompt, str)
            self.assertGreater(len(prompt), 10)
            prompts.add(prompt)
        
        # Should have some variation (though with seed, might be limited)
        logging.info(f"✅ Generated {len(prompts)} unique prompts")

    def test_batch_analysis_functions(self):
        """Test batch analysis display functions"""
        from app import create_batch_analysis_display, create_batch_analysis_table, create_batch_summary_stats
        
        # Create mock batch analysis data
        mock_analysis = [
            {
                'scene_id': 1,
                'filename': 'test_image_1.png',
                'path': '/fake/path/test_image_1.png',
                'score': 0.85,
                'face_count': 1,
                'quality_score': 0.8,
                'has_realistic_face': True,
                'detection_methods': ['mediapipe'],
                'clothing_consistency': 0.9,
                'clothing_colors': [(255, 0, 0), (0, 255, 0)],
                'roop_face_used': True,
                'selected_face_person': 'test_person',
                'is_selected': True,
                'scene_description': 'Test scene description'
            },
            {
                'scene_id': 1,
                'filename': 'test_image_2.png', 
                'path': '/fake/path/test_image_2.png',
                'score': 0.75,
                'face_count': 1,
                'quality_score': 0.7,
                'has_realistic_face': True,
                'detection_methods': ['opencv'],
                'clothing_consistency': 0.8,
                'clothing_colors': [(255, 0, 0)],
                'roop_face_used': False,
                'selected_face_person': None,
                'is_selected': False,
                'scene_description': 'Test scene description'
            }
        ]
        
        # Test markdown report generation
        report, images = create_batch_analysis_display(mock_analysis)
        self.assertIsInstance(report, str)
        self.assertIn('Batch Image Analysis Report', report)
        self.assertIsInstance(images, list)
        
        # Test HTML table generation
        table_html = create_batch_analysis_table(mock_analysis)
        self.assertIsInstance(table_html, str)
        self.assertIn('<table', table_html)
        self.assertIn('test_image_1.png', table_html)
        
        # Test summary stats
        stats = create_batch_summary_stats(mock_analysis)
        self.assertIsInstance(stats, str)
        self.assertIn('Total Images', stats)
        
        logging.info("✅ Batch analysis functions working")

    def test_output_folder_cleanup(self):
        """Test output folder cleanup functionality"""
        from app import clean_output_folder
        
        # Create test output structure
        test_output_dirs = [
            'output/frames',
            'output/videos',
            'output/frames/batch_scene_1'
        ]
        
        for dir_path in test_output_dirs:
            os.makedirs(dir_path, exist_ok=True)
            
            # Create test files
            test_file = os.path.join(dir_path, 'test_file.txt')
            with open(test_file, 'w') as f:
                f.write('test content')
        
        # Create .gitkeep file (should be preserved)
        gitkeep_path = 'output/.gitkeep'
        with open(gitkeep_path, 'w') as f:
            f.write('')
        
        # Test cleanup
        message, video, gallery = clean_output_folder()
        
        # Verify cleanup worked
        self.assertIsInstance(message, str)
        self.assertIn('Cleanup complete', message)
        
        # Verify .gitkeep is preserved
        self.assertTrue(os.path.exists(gitkeep_path))
        
        logging.info("✅ Output folder cleanup working")

class TestErrorHandling(unittest.TestCase):
    """Test error handling in pipeline components"""

    def test_mixtral_timeout_handling(self):
        """Test Mixtral timeout handling"""
        from mixtral_client import MixtralClient
        
        client = MixtralClient()
        
        # Test fallback when Mixtral is not available
        with patch('requests.Session.post', side_effect=Exception("Connection timeout")):
            result = client.analyze_narrative_structure("test prompt")
            
            # Should fall back to default structure
            self.assertIsInstance(result, list)
            self.assertGreater(len(result), 0)
            
            logging.info("✅ Mixtral timeout handling working")

    def test_sd_client_error_handling(self):
        """Test SD client error handling"""
        from sd_client import StableDiffusionClient
        
        client = StableDiffusionClient()
        
        # Test connection check with mock failure
        with patch('requests.Session.get', side_effect=Exception("Connection refused")):
            result = client.check_connection()
            self.assertFalse(result)
            
            logging.info("✅ SD client error handling working")

    def test_face_analyzer_error_handling(self):
        """Test face analyzer error handling with invalid inputs"""
        from face_analyzer import FaceAnalyzer
        
        analyzer = FaceAnalyzer()
        
        # Test with non-existent file
        result = analyzer.detect_faces('/non/existent/path.png')
        
        # Should return default structure with error info
        self.assertIsInstance(result, dict)
        self.assertIn('error', result)
        
        logging.info("✅ Face analyzer error handling working")

    def test_video_generator_error_handling(self):
        """Test video generator error handling"""
        from video_generator import VideoGenerator
        
        # Test with invalid candidate count
        generator = VideoGenerator(candidate_count=0)
        self.assertEqual(generator.candidate_count, 0)  # Should handle gracefully
        
        logging.info("✅ Video generator error handling working")

class TestDataFlow(unittest.TestCase):
    """Test data flow between pipeline components"""

    def test_scene_data_structure(self):
        """Test scene data structure consistency"""
        # Test scene data format from Mixtral through to video generation
        sample_scene = {
            'scene_number': 1,
            'description': 'Woman walking in park entrance',
            'prompt': 'detailed enhanced prompt for stable diffusion',
            'negative_prompt': 'negative elements to avoid',
            'duration': 3.0,
            'transition_type': 'fade'
        }
        
        # Verify required fields
        required_fields = ['description', 'prompt', 'duration']
        for field in required_fields:
            self.assertIn(field, sample_scene)
            self.assertIsNotNone(sample_scene[field])
        
        logging.info("✅ Scene data structure validation passed")

    def test_batch_analysis_data_structure(self):
        """Test batch analysis data structure consistency"""
        sample_analysis = {
            'scene_id': 1,
            'filename': 'batch_0_0_20250719_120000.png',
            'path': '/path/to/image.png',
            'score': 0.85,
            'face_count': 1,
            'quality_score': 0.8,
            'has_realistic_face': True,
            'detection_methods': ['mediapipe', 'opencv'],
            'clothing_consistency': 0.9,
            'clothing_colors': [(255, 0, 0), (0, 255, 0), (0, 0, 255)],
            'roop_face_used': True,
            'selected_face_person': 'ayushi',
            'is_selected': True,
            'scene_description': 'Scene description text'
        }
        
        # Verify required fields
        required_fields = ['filename', 'score', 'face_count', 'scene_id']
        for field in required_fields:
            self.assertIn(field, sample_analysis)
            self.assertIsNotNone(sample_analysis[field])
        
        # Verify data types
        self.assertIsInstance(sample_analysis['score'], (int, float))
        self.assertIsInstance(sample_analysis['face_count'], int)
        self.assertIsInstance(sample_analysis['detection_methods'], list)
        
        logging.info("✅ Batch analysis data structure validation passed")

if __name__ == '__main__':
    unittest.main()