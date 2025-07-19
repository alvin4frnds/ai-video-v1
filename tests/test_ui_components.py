"""
Test UI components and Gradio interface functionality
"""

import unittest
import logging
import os
import sys
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestGradioComponents(unittest.TestCase):
    """Test Gradio UI components and interface"""

    def test_gradio_import_and_basic_components(self):
        """Test Gradio imports and basic component creation"""
        try:
            import gradio as gr
            
            # Test basic component creation
            textbox = gr.Textbox(label="Test Textbox")
            self.assertIsNotNone(textbox)
            
            button = gr.Button("Test Button")
            self.assertIsNotNone(button)
            
            number = gr.Number(label="Test Number", value=2)
            self.assertIsNotNone(number)
            
            checkbox = gr.Checkbox(label="Test Checkbox", value=False)
            self.assertIsNotNone(checkbox)
            
            video = gr.Video(label="Test Video")
            self.assertIsNotNone(video)
            
            gallery = gr.Gallery(label="Test Gallery")
            self.assertIsNotNone(gallery)
            
            html = gr.HTML(value="<p>Test HTML</p>")
            self.assertIsNotNone(html)
            
            logging.info("✅ Gradio component creation working")
            
        except ImportError as e:
            self.fail(f"Failed to import Gradio: {e}")

    def test_app_ui_structure(self):
        """Test app.py UI structure and components"""
        try:
            # Import app.py (this will test if Gradio interface can be created)
            import app
            
            # Test that demo object exists
            self.assertTrue(hasattr(app, 'demo'))
            
            # Test key UI functions
            ui_functions = [
                'generate_random_test_prompt',
                'create_still_preview', 
                'create_batch_analysis_display',
                'create_batch_analysis_table',
                'create_batch_summary_stats',
                'clean_output_folder'
            ]
            
            for func_name in ui_functions:
                with self.subTest(function=func_name):
                    self.assertTrue(hasattr(app, func_name),
                                  f"UI function missing: {func_name}")
            
            logging.info("✅ App UI structure validation passed")
            
        except ImportError as e:
            self.fail(f"Failed to import app for UI testing: {e}")

    def test_ui_handler_functions(self):
        """Test UI event handler functions"""
        import app
        
        # Test random prompt generation
        prompt = app.generate_random_test_prompt()
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 5)
        
        # Test still preview creation
        mock_images = [
            {'path': '/fake/path1.png', 'description': 'test1'},
            {'path': '/fake/path2.png', 'description': 'test2'}
        ]
        preview = app.create_still_preview(mock_images)
        self.assertIsInstance(preview, list)
        
        # Test empty batch analysis
        report, images = app.create_batch_analysis_display([])
        self.assertIn("No batch analysis data", report)
        self.assertEqual(images, [])
        
        # Test empty batch table
        table_html = app.create_batch_analysis_table([])
        self.assertIn("No batch analysis data", table_html)
        
        # Test empty batch stats
        stats = app.create_batch_summary_stats([])
        self.assertEqual(stats, "No data")
        
        logging.info("✅ UI handler functions working")

class TestUIDataFormatting(unittest.TestCase):
    """Test UI data formatting and display functions"""

    def setUp(self):
        # Sample data for testing
        self.sample_batch_analysis = [
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
                'scene_description': 'Test scene 1',
                'faces': [
                    {'confidence': 0.95, 'method': 'mediapipe', 'area': 1000}
                ]
            },
            {
                'scene_id': 2,
                'filename': 'test_image_2.png',
                'path': '/fake/path/test_image_2.png', 
                'score': 0.75,
                'face_count': 1,
                'quality_score': 0.7,
                'has_realistic_face': False,
                'detection_methods': ['opencv'],
                'clothing_consistency': 0.6,
                'clothing_colors': [(0, 0, 255)],
                'roop_face_used': False,
                'selected_face_person': None,
                'is_selected': False,
                'scene_description': 'Test scene 2',
                'error': 'Test error message'
            }
        ]

    def test_batch_analysis_report_generation(self):
        """Test batch analysis markdown report generation"""
        import app
        
        report, image_paths = app.create_batch_analysis_display(self.sample_batch_analysis)
        
        # Test report structure
        self.assertIsInstance(report, str)
        self.assertIn('Batch Image Analysis Report', report)
        self.assertIn('Scene 1', report)
        self.assertIn('Scene 2', report)
        self.assertIn('test_image_1.png', report)
        self.assertIn('test_image_2.png', report)
        
        # Test scoring display
        self.assertIn('0.85', report)  # Score
        self.assertIn('0.9', report)   # Clothing consistency
        
        # Test status indicators
        self.assertIn('✅ Yes', report)  # Realistic face
        self.assertIn('❌ No', report)   # Non-realistic face
        self.assertIn('✅ Used', report) # Roop used
        self.assertIn('❌ Not used', report) # Roop not used
        
        # Test error handling
        self.assertIn('Test error message', report)
        
        # Test image paths (should be empty for fake paths)
        self.assertIsInstance(image_paths, list)
        
        logging.info("✅ Batch analysis report generation working")

    def test_batch_analysis_table_generation(self):
        """Test batch analysis HTML table generation"""
        import app
        
        table_html = app.create_batch_analysis_table(self.sample_batch_analysis)
        
        # Test HTML structure
        self.assertIsInstance(table_html, str)
        self.assertIn('<table', table_html)
        self.assertIn('<thead>', table_html)
        self.assertIn('<tbody>', table_html)
        self.assertIn('</table>', table_html)
        
        # Test CSS styling
        self.assertIn('batch-table', table_html)
        self.assertIn('rank-1', table_html)
        self.assertIn('selected', table_html)
        
        # Test data content
        self.assertIn('test_image_1.png', table_html)
        self.assertIn('test_image_2.png', table_html)
        self.assertIn('0.850', table_html)  # Formatted score
        self.assertIn('0.750', table_html)
        
        # Test thumbnails (even if fake paths)
        self.assertIn('thumbnail', table_html)
        
        # Test ranking
        self.assertIn('🥇', table_html)  # First place
        self.assertIn('⭐', table_html)  # Selected
        
        # Test consistency icons
        self.assertIn('✅', table_html)  # High consistency
        self.assertIn('⚠️', table_html)   # Medium consistency
        
        logging.info("✅ Batch analysis table generation working")

    def test_batch_summary_stats_generation(self):
        """Test batch summary statistics generation"""
        import app
        
        stats = app.create_batch_summary_stats(self.sample_batch_analysis)
        
        # Test stats structure
        self.assertIsInstance(stats, str)
        self.assertIn('Batch Analysis Summary', stats)
        
        # Test statistics content
        self.assertIn('Total Images**: 2', stats)
        self.assertIn('Total Faces Detected**: 2', stats)
        self.assertIn('Realistic Faces**: 1/2', stats)
        self.assertIn('50.0%', stats)  # Percentage
        
        # Test Roop statistics
        self.assertIn('Roop Face Swap Used**: 1/2', stats)
        self.assertIn('test_person', stats)
        
        # Test detection methods
        self.assertIn('mediapipe', stats)
        self.assertIn('opencv', stats)
        
        logging.info("✅ Batch summary stats generation working")

class TestUIInteractions(unittest.TestCase):
    """Test UI interactions and event handling"""

    @patch('app.generate_video_pipeline')
    def test_generation_handler_mock(self, mock_pipeline):
        """Test generation handler with mocked pipeline"""
        import app
        
        # Mock pipeline return
        mock_pipeline.return_value = (
            '/fake/video.mp4',     # video_path
            [{'path': '/fake/frame1.png'}],  # generated_images
            [],                    # batch_analysis
            'Success'              # logs
        )
        
        # Test generation handler
        result = app.handle_generation(
            prompt="test prompt",
            candidate_count=2,
            clean_before_run=False
        )
        
        # Verify handler returns expected structure
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 6)  # video, gallery, report, images, stats, table
        
        # Verify mock was called with correct parameters
        mock_pipeline.assert_called_once_with("test prompt", 2, False)
        
        logging.info("✅ Generation handler mock test working")

    @patch('app.generate_video_pipeline')
    def test_random_test_handler_mock(self, mock_pipeline):
        """Test random test handler with mocked pipeline"""
        import app
        
        # Mock pipeline return
        mock_pipeline.return_value = (
            '/fake/video.mp4',
            [{'path': '/fake/frame1.png'}],
            [],
            'Success'
        )
        
        # Test random test handler
        result = app.handle_random_test(
            candidate_count=3,
            clean_before_run=True
        )
        
        # Verify handler returns expected structure
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 7)  # prompt + video, gallery, report, images, stats, table
        
        # Verify random prompt was generated
        random_prompt = result[0]
        self.assertIsInstance(random_prompt, str)
        self.assertGreater(len(random_prompt), 5)
        
        logging.info("✅ Random test handler mock test working")

    def test_clean_handler(self):
        """Test clean button handler"""
        import app
        
        # Test clean handler
        result = app.handle_clean()
        
        # Verify handler returns expected structure
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 7)  # message, video, gallery, report, images, stats, table
        
        # Verify message is returned
        message = result[0]
        self.assertIsInstance(message, str)
        
        # Verify reset values
        self.assertIsNone(result[1])  # video should be None
        self.assertEqual(result[2], [])  # gallery should be empty
        
        logging.info("✅ Clean handler test working")

class TestUICSS(unittest.TestCase):
    """Test UI CSS and styling"""

    def test_css_structure(self):
        """Test CSS structure in app.py"""
        import app
        
        # Test that CSS variable exists
        self.assertTrue(hasattr(app, 'css'))
        
        css_content = app.css
        self.assertIsInstance(css_content, str)
        
        # Test key CSS elements
        css_checks = [
            'font-family',
            '-apple-system',
            'BlinkMacSystemFont',
            'Segoe UI',
            'sans-serif'
        ]
        
        for css_element in css_checks:
            with self.subTest(css_element=css_element):
                self.assertIn(css_element, css_content)
        
        logging.info("✅ CSS structure validation passed")

    def test_table_css_styling(self):
        """Test table CSS styling for contrast and readability"""
        import app
        
        # Generate sample table to check CSS application
        sample_data = [
            {
                'scene_id': 1,
                'filename': 'test.png',
                'path': '/fake/path.png',
                'score': 0.9,
                'face_count': 1,
                'quality_score': 0.8,
                'has_realistic_face': True,
                'detection_methods': ['test'],
                'clothing_consistency': 0.8,
                'roop_face_used': False,
                'selected_face_person': None,
                'is_selected': True,
                'scene_description': 'Test scene'
            }
        ]
        
        table_html = app.create_batch_analysis_table(sample_data)
        
        # Test CSS classes for ranking and selection
        css_classes = [
            'batch-table',
            'rank-1',
            'selected',
            'score-high',
            'consistency-icon'
        ]
        
        for css_class in css_classes:
            with self.subTest(css_class=css_class):
                self.assertIn(css_class, table_html)
        
        # Test color styling for contrast
        self.assertIn('#8a6914', table_html)  # Dark gold text
        self.assertIn('#6f42c1', table_html)  # Dark purple text
        self.assertIn('#155724', table_html)  # Dark green text
        
        logging.info("✅ Table CSS styling validation passed")

if __name__ == '__main__':
    unittest.main()