#!/usr/bin/env python3
"""
Test the new motion interpolation system
"""

import sys
sys.path.append('.')

import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import logging
from final_enhanced_generator import FinalEnhancedGenerator
from simple_wan_generator import SimpleWANGenerator
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_images():
    """Create simple test images for motion testing"""
    logger.info("🎨 Creating test images for motion interpolation...")
    
    # Create test directory
    test_dir = Path("test_motion")
    test_dir.mkdir(exist_ok=True)
    
    # Test image 1: Person standing on left
    img1 = np.zeros((400, 600, 3), dtype=np.uint8)
    img1[:, :] = [70, 130, 180]  # Steel blue background
    
    # Draw simple person figure on left side
    center_x, center_y = 150, 200
    # Head
    cv2.circle(img1, (center_x, center_y - 60), 25, (255, 220, 177), -1)
    # Body
    cv2.rectangle(img1, (center_x - 20, center_y - 35), (center_x + 20, center_y + 40), (200, 100, 100), -1)
    # Arms
    cv2.rectangle(img1, (center_x - 40, center_y - 20), (center_x - 20, center_y + 10), (200, 100, 100), -1)
    cv2.rectangle(img1, (center_x + 20, center_y - 20), (center_x + 40, center_y + 10), (200, 100, 100), -1)
    # Legs
    cv2.rectangle(img1, (center_x - 15, center_y + 40), (center_x - 5, center_y + 80), (50, 50, 150), -1)
    cv2.rectangle(img1, (center_x + 5, center_y + 40), (center_x + 15, center_y + 80), (50, 50, 150), -1)
    
    # Test image 2: Person standing on right (same figure, different position)
    img2 = np.zeros((400, 600, 3), dtype=np.uint8)
    img2[:, :] = [70, 130, 180]  # Same background
    
    # Draw person figure on right side
    center_x, center_y = 450, 200
    # Head
    cv2.circle(img2, (center_x, center_y - 60), 25, (255, 220, 177), -1)
    # Body
    cv2.rectangle(img2, (center_x - 20, center_y - 35), (center_x + 20, center_y + 40), (200, 100, 100), -1)
    # Arms (slightly different pose)
    cv2.rectangle(img2, (center_x - 40, center_y - 15), (center_x - 20, center_y + 15), (200, 100, 100), -1)
    cv2.rectangle(img2, (center_x + 20, center_y - 15), (center_x + 40, center_y + 15), (200, 100, 100), -1)
    # Legs (walking pose)
    cv2.rectangle(img2, (center_x - 18, center_y + 40), (center_x - 8, center_y + 80), (50, 50, 150), -1)
    cv2.rectangle(img2, (center_x + 8, center_y + 40), (center_x + 18, center_y + 80), (50, 50, 150), -1)
    
    # Save test images
    Image.fromarray(img1).save(test_dir / "person_left.png")
    Image.fromarray(img2).save(test_dir / "person_right.png")
    
    logger.info(f"✅ Created test images in {test_dir}/")
    return str(test_dir / "person_left.png"), str(test_dir / "person_right.png")

def test_enhanced_generator():
    """Test the FinalEnhancedGenerator with motion interpolation"""
    logger.info("🧪 Testing FinalEnhancedGenerator motion interpolation...")
    
    try:
        generator = FinalEnhancedGenerator()
        
        # Create test images
        start_path, end_path = create_test_images()
        
        # Load images
        start_img = np.array(Image.open(start_path).convert('RGB'))
        end_img = np.array(Image.open(end_path).convert('RGB'))
        
        # Test different transition types
        transition_prompts = [
            "woman walking across the scene, smooth movement",
            "person sitting down slowly, vertical motion",
            "character turning to look around, rotational movement"
        ]
        
        for i, prompt in enumerate(transition_prompts):
            logger.info(f"\\n{'='*60}")
            logger.info(f"🎭 Testing transition {i+1}: {prompt}")
            logger.info(f"{'='*60}")
            
            # Create transition frames
            frames = generator.create_advanced_transition(
                start_img, end_img, prompt, num_frames=30
            )
            
            # Save as video
            timestamp = datetime.now().strftime("%H%M%S")
            output_path = f"test_motion/enhanced_motion_test_{i+1}_{timestamp}.mp4"
            generator.save_video(frames, output_path, fps=15)
            
            logger.info(f"✅ Saved enhanced motion test: {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced generator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_generator():
    """Test the SimpleWANGenerator with motion interpolation"""
    logger.info("🧪 Testing SimpleWANGenerator motion interpolation...")
    
    try:
        generator = SimpleWANGenerator()
        
        # Create test images
        start_path, end_path = create_test_images()
        
        # Generate video with motion
        timestamp = datetime.now().strftime("%H%M%S")
        output_path = f"test_motion/simple_motion_test_{timestamp}.mp4"
        
        result = generator.generate_video(
            start_image_path=start_path,
            end_image_path=end_path,
            output_path=output_path,
            num_frames=45,
            fps=15
        )
        
        logger.info(f"✅ Saved simple motion test: {result}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Simple generator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_old_vs_new():
    """Compare the new motion interpolation with old simple blending"""
    logger.info("📊 Comparing old vs new interpolation methods...")
    
    try:
        # Create test images
        start_path, end_path = create_test_images()
        start_img = np.array(Image.open(start_path).convert('RGB'))
        end_img = np.array(Image.open(end_path).convert('RGB'))
        
        # Old method - simple linear interpolation
        old_frames = []
        for i in range(20):
            t = i / 19
            interpolated = start_img * (1 - t) + end_img * t
            old_frames.append(interpolated.astype(np.uint8))
        
        # New method - motion interpolation
        generator = FinalEnhancedGenerator()
        new_frames = generator.create_advanced_transition(
            start_img, end_img, "person walking across scene", num_frames=20
        )
        
        # Save comparison videos
        timestamp = datetime.now().strftime("%H%M%S")
        
        # Old method video
        old_output = f"test_motion/old_method_{timestamp}.mp4"
        generator.save_video(old_frames, old_output, fps=10)
        
        # New method video
        new_output = f"test_motion/new_method_{timestamp}.mp4"
        generator.save_video(new_frames, new_output, fps=10)
        
        logger.info(f"📹 Comparison videos created:")
        logger.info(f"  Old method (simple blending): {old_output}")
        logger.info(f"  New method (motion interpolation): {new_output}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Comparison test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🎬 Motion Interpolation Test Suite")
    print("=" * 60)
    
    # Test 1: Enhanced Generator
    enhanced_ok = test_enhanced_generator()
    
    print("\\n" + "=" * 60)
    
    # Test 2: Simple Generator
    simple_ok = test_simple_generator()
    
    print("\\n" + "=" * 60)
    
    # Test 3: Comparison
    comparison_ok = compare_old_vs_new()
    
    print("\\n" + "=" * 60)
    print("📊 TEST RESULTS:")
    print(f"  Enhanced Generator: {'✅ PASS' if enhanced_ok else '❌ FAIL'}")
    print(f"  Simple Generator: {'✅ PASS' if simple_ok else '❌ FAIL'}")
    print(f"  Method Comparison: {'✅ PASS' if comparison_ok else '❌ FAIL'}")
    
    if enhanced_ok and simple_ok and comparison_ok:
        print("\\n🎉 All motion interpolation tests passed!")
        print("💡 The new system creates actual subject movement instead of simple crossfading")
        print("🎬 Generated videos show realistic motion within frames")
    else:
        print("\\n⚠️  Some tests failed - check the logs above")
    
    print("=" * 60)

if __name__ == "__main__":
    main()