"""
Test all package imports and dependencies
"""

import unittest
import sys
import importlib
import logging

class TestImports(unittest.TestCase):
    """Test all required package imports"""

    def setUp(self):
        self.required_packages = [
            'gradio',
            'requests', 
            'json',
            'logging',
            'cv2',
            'numpy',
            'PIL',
            'datetime',
            'os',
            'base64',
            'io',
            'random',
            'time',
            'subprocess',
            'shutil',
            'threading'
        ]
        
        self.optional_packages = [
            'mediapipe',
            'deepface',
            'sklearn'
        ]

    def test_core_python_modules(self):
        """Test core Python standard library modules"""
        core_modules = ['json', 'logging', 'datetime', 'os', 'base64', 'io', 'random', 'time', 'subprocess', 'shutil', 'threading']
        
        for module_name in core_modules:
            with self.subTest(module=module_name):
                try:
                    importlib.import_module(module_name)
                    logging.info(f"✅ {module_name} imported successfully")
                except ImportError as e:
                    self.fail(f"Failed to import core module {module_name}: {e}")

    def test_gradio_import(self):
        """Test Gradio import and version"""
        try:
            import gradio as gr
            logging.info(f"✅ Gradio imported successfully (version: {gr.__version__})")
            
            # Test basic Gradio functionality
            self.assertTrue(hasattr(gr, 'Blocks'))
            self.assertTrue(hasattr(gr, 'Textbox'))
            self.assertTrue(hasattr(gr, 'Button'))
            
        except ImportError as e:
            self.fail(f"Failed to import Gradio: {e}")

    def test_opencv_import(self):
        """Test OpenCV import and basic functionality"""
        try:
            import cv2
            logging.info(f"✅ OpenCV imported successfully (version: {cv2.__version__})")
            
            # Test basic OpenCV functionality
            self.assertTrue(hasattr(cv2, 'VideoWriter'))
            self.assertTrue(hasattr(cv2, 'VideoWriter_fourcc'))
            
        except ImportError as e:
            self.fail(f"Failed to import OpenCV: {e}")

    def test_numpy_import(self):
        """Test NumPy import and basic functionality"""
        try:
            import numpy as np
            logging.info(f"✅ NumPy imported successfully (version: {np.__version__})")
            
            # Test basic NumPy functionality
            test_array = np.array([1, 2, 3])
            self.assertEqual(len(test_array), 3)
            
        except ImportError as e:
            self.fail(f"Failed to import NumPy: {e}")

    def test_pil_import(self):
        """Test PIL/Pillow import and basic functionality"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            import PIL
            logging.info(f"✅ PIL/Pillow imported successfully (version: {PIL.__version__})")
            
            # Test basic PIL functionality
            self.assertTrue(hasattr(Image, 'open'))
            self.assertTrue(hasattr(Image, 'new'))
            
        except ImportError as e:
            self.fail(f"Failed to import PIL/Pillow: {e}")

    def test_requests_import(self):
        """Test requests import and basic functionality"""
        try:
            import requests
            logging.info(f"✅ Requests imported successfully (version: {requests.__version__})")
            
            # Test basic requests functionality
            self.assertTrue(hasattr(requests, 'get'))
            self.assertTrue(hasattr(requests, 'post'))
            self.assertTrue(hasattr(requests, 'Session'))
            
        except ImportError as e:
            self.fail(f"Failed to import requests: {e}")

    def test_optional_packages(self):
        """Test optional packages (may not be critical for basic functionality)"""
        for package_name in self.optional_packages:
            with self.subTest(package=package_name):
                try:
                    if package_name == 'sklearn':
                        from sklearn.cluster import KMeans
                        logging.info(f"✅ {package_name} imported successfully")
                    else:
                        importlib.import_module(package_name)
                        logging.info(f"✅ {package_name} imported successfully")
                except ImportError as e:
                    logging.warning(f"⚠️ Optional package {package_name} not available: {e}")

    def test_project_modules_import(self):
        """Test project-specific module imports"""
        project_modules = [
            'mixtral_client',
            'sd_client', 
            'face_analyzer',
            'face_selector',
            'video_generator',
            'app'
        ]
        
        for module_name in project_modules:
            with self.subTest(module=module_name):
                try:
                    importlib.import_module(module_name)
                    logging.info(f"✅ Project module {module_name} imported successfully")
                except ImportError as e:
                    self.fail(f"Failed to import project module {module_name}: {e}")

if __name__ == '__main__':
    unittest.main()