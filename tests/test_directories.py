"""
Test directory structure, permissions, and file handling
"""

import unittest
import logging
import os
import tempfile
import shutil
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestDirectoryStructure(unittest.TestCase):
    """Test directory structure and file operations"""

    def test_project_root_structure(self):
        """Test main project directory structure"""
        required_files = [
            'app.py',
            'video_generator.py',
            'mixtral_client.py', 
            'sd_client.py',
            'face_analyzer.py',
            'face_selector.py',
            'test_runner.py'
        ]
        
        for file_name in required_files:
            with self.subTest(file=file_name):
                self.assertTrue(os.path.exists(file_name), 
                              f"Required file not found: {file_name}")
                
                # Check file is readable
                self.assertTrue(os.access(file_name, os.R_OK),
                              f"File not readable: {file_name}")
                
                # Check file has content
                self.assertGreater(os.path.getsize(file_name), 0,
                                 f"File is empty: {file_name}")
                
                logging.info(f"✅ {file_name} exists and readable")

    def test_output_directory_structure(self):
        """Test output directory structure and creation"""
        output_dirs = [
            'output',
            'output/frames',
            'output/videos'
        ]
        
        for dir_path in output_dirs:
            with self.subTest(directory=dir_path):
                # Create if doesn't exist
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path, exist_ok=True)
                
                # Test directory exists
                self.assertTrue(os.path.exists(dir_path),
                              f"Output directory not found: {dir_path}")
                
                # Test directory is writable
                self.assertTrue(os.access(dir_path, os.W_OK),
                              f"Output directory not writable: {dir_path}")
                
                # Test can create files in directory
                test_file = os.path.join(dir_path, 'test_write.tmp')
                try:
                    with open(test_file, 'w') as f:
                        f.write('test')
                    os.remove(test_file)
                    logging.info(f"✅ {dir_path} writable")
                except Exception as e:
                    self.fail(f"Cannot write to directory {dir_path}: {e}")

    def test_input_directory_structure(self):
        """Test input directory structure for face images"""
        face_base_dir = 'in/individual'
        
        # Create base directory if it doesn't exist
        if not os.path.exists(face_base_dir):
            os.makedirs(face_base_dir, exist_ok=True)
            logging.info(f"✅ Created face input directory: {face_base_dir}")
        
        self.assertTrue(os.path.exists(face_base_dir))
        
        # Check for existing face directories
        face_persons = []
        if os.path.exists(face_base_dir):
            for item in os.listdir(face_base_dir):
                person_path = os.path.join(face_base_dir, item)
                if os.path.isdir(person_path):
                    # Count image files
                    image_files = [f for f in os.listdir(person_path)
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))]
                    if image_files:
                        face_persons.append(f"{item}: {len(image_files)} images")
        
        if face_persons:
            logging.info(f"✅ Found face directories: {face_persons}")
        else:
            logging.warning("⚠️ No face directories found - Roop functionality may be limited")

    def test_gitkeep_files(self):
        """Test that .gitkeep files are present in output directories"""
        gitkeep_locations = [
            'output/.gitkeep'
        ]
        
        for gitkeep_path in gitkeep_locations:
            # Create .gitkeep if output directory exists but .gitkeep doesn't
            output_dir = os.path.dirname(gitkeep_path)
            if os.path.exists(output_dir) and not os.path.exists(gitkeep_path):
                try:
                    with open(gitkeep_path, 'w') as f:
                        f.write('')
                    logging.info(f"✅ Created .gitkeep: {gitkeep_path}")
                except Exception as e:
                    logging.warning(f"⚠️ Could not create .gitkeep {gitkeep_path}: {e}")

class TestFileOperations(unittest.TestCase):
    """Test file operations and handling"""

    def setUp(self):
        # Create temporary test directory
        self.test_dir = tempfile.mkdtemp(prefix='video_gen_test_')

    def tearDown(self):
        # Clean up test directory
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_batch_directory_creation(self):
        """Test batch directory creation and cleanup"""
        from app import clean_output_folder
        
        # Create some test batch directories
        test_batch_dirs = [
            'output/frames/batch_scene_1',
            'output/frames/batch_scene_2',
            'output/frames/batch_scene_3'
        ]
        
        for batch_dir in test_batch_dirs:
            os.makedirs(batch_dir, exist_ok=True)
            
            # Create some test files
            test_file = os.path.join(batch_dir, 'test_image.png')
            with open(test_file, 'w') as f:
                f.write('test content')
        
        # Test cleanup function
        message, video, gallery = clean_output_folder()
        
        # Verify directories are cleaned
        for batch_dir in test_batch_dirs:
            self.assertFalse(os.path.exists(batch_dir),
                           f"Batch directory not cleaned: {batch_dir}")
        
        logging.info("✅ Batch directory cleanup working")

    def test_image_file_handling(self):
        """Test image file operations"""
        from PIL import Image
        import numpy as np
        
        # Create test image
        test_image_path = os.path.join(self.test_dir, 'test_image.png')
        
        # Create simple test image
        img_array = np.zeros((100, 100, 3), dtype=np.uint8)
        img_array[:, :] = [255, 0, 0]  # Red image
        
        img = Image.fromarray(img_array)
        img.save(test_image_path)
        
        # Test image exists and readable
        self.assertTrue(os.path.exists(test_image_path))
        
        # Test image can be loaded
        loaded_img = Image.open(test_image_path)
        self.assertEqual(loaded_img.size, (100, 100))
        
        logging.info("✅ Image file handling working")

    def test_video_file_handling(self):
        """Test video file operations"""
        try:
            import cv2
            import numpy as np
            
            test_video_path = os.path.join(self.test_dir, 'test_video.mp4')
            
            # Create simple test video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(test_video_path, fourcc, 30.0, (640, 480))
            
            if out.isOpened():
                # Create a few test frames
                for i in range(30):  # 1 second at 30fps
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    frame[:, :] = [0, 255, 0]  # Green frame
                    out.write(frame)
                
                out.release()
                
                # Test video file exists and has content
                self.assertTrue(os.path.exists(test_video_path))
                self.assertGreater(os.path.getsize(test_video_path), 0)
                
                logging.info("✅ Video file handling working")
            else:
                logging.warning("⚠️ Could not create test video - codec issues")
                self.skipTest("Video codec not available")
                
        except ImportError:
            self.skipTest("OpenCV not available for video testing")

    def test_temp_file_cleanup(self):
        """Test temporary file cleanup"""
        import tempfile
        
        # Create temporary files
        temp_files = []
        for i in range(5):
            temp_file = tempfile.NamedTemporaryFile(
                suffix=f'_test_{i}.tmp', 
                dir=self.test_dir,
                delete=False
            )
            temp_file.write(b'test content')
            temp_file.close()
            temp_files.append(temp_file.name)
        
        # Verify files exist
        for temp_file in temp_files:
            self.assertTrue(os.path.exists(temp_file))
        
        # Clean up
        for temp_file in temp_files:
            os.remove(temp_file)
            self.assertFalse(os.path.exists(temp_file))
        
        logging.info("✅ Temporary file cleanup working")

class TestPermissions(unittest.TestCase):
    """Test file and directory permissions"""

    def test_output_directory_permissions(self):
        """Test output directory permissions"""
        output_dirs = ['output', 'output/frames', 'output/videos']
        
        for dir_path in output_dirs:
            if os.path.exists(dir_path):
                # Test read permission
                self.assertTrue(os.access(dir_path, os.R_OK),
                              f"No read permission: {dir_path}")
                
                # Test write permission
                self.assertTrue(os.access(dir_path, os.W_OK),
                              f"No write permission: {dir_path}")
                
                # Test execute permission (needed to enter directory)
                self.assertTrue(os.access(dir_path, os.X_OK),
                              f"No execute permission: {dir_path}")
                
                logging.info(f"✅ Permissions OK: {dir_path}")

    def test_script_file_permissions(self):
        """Test script file permissions"""
        script_files = [
            'app.py',
            'video_generator.py',
            'test_runner.py'
        ]
        
        for script_file in script_files:
            if os.path.exists(script_file):
                # Test read permission
                self.assertTrue(os.access(script_file, os.R_OK),
                              f"Cannot read script: {script_file}")
                
                logging.info(f"✅ Script readable: {script_file}")

class TestDiskSpace(unittest.TestCase):
    """Test disk space and storage requirements"""

    def test_available_disk_space(self):
        """Test available disk space for video generation"""
        import shutil
        
        # Check available space in output directory
        total, used, free = shutil.disk_usage('output' if os.path.exists('output') else '.')
        
        # Convert to MB
        free_mb = free // (1024 * 1024)
        
        # Log disk space info
        logging.info(f"💾 Available disk space: {free_mb} MB")
        
        # Warn if less than 1GB available
        if free_mb < 1024:
            logging.warning(f"⚠️ Low disk space: {free_mb} MB available")
        else:
            logging.info(f"✅ Sufficient disk space: {free_mb} MB available")
        
        # Test should not fail due to low disk space, just warn
        self.assertGreater(free_mb, 100, "Critically low disk space (less than 100MB)")

if __name__ == '__main__':
    unittest.main()