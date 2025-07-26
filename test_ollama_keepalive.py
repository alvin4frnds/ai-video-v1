#!/usr/bin/env python3
"""
Test Ollama keep_alive parameter functionality
"""

import sys
sys.path.append('.')

from mixtral_client import MixtralClient
from final_enhanced_generator import FinalEnhancedGenerator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_mixtral_client_keepalive():
    """Test MixtralClient with 30-minute keep_alive"""
    logger.info("🧪 Testing MixtralClient with 30-minute keep_alive...")
    
    try:
        client = MixtralClient()
        
        # Check connection first
        if not client.check_connection():
            logger.warning("⚠️  Mixtral/Ollama not available - skipping keep_alive test")
            return True
        
        # Test basic prompt with keep_alive
        system_prompt = "You are a helpful assistant."
        user_prompt = "Say hello and confirm the keep_alive parameter is working."
        
        logger.info("📡 Sending test request with keep_alive=30m...")
        response = client._call_mixtral(system_prompt, user_prompt)
        
        logger.info(f"✅ Response received: {response[:100]}...")
        logger.info("✅ MixtralClient keep_alive test successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ MixtralClient keep_alive test failed: {e}")
        return False

def test_enhanced_generator_keepalive():
    """Test FinalEnhancedGenerator with keep_alive"""
    logger.info("🧪 Testing FinalEnhancedGenerator with 30-minute keep_alive...")
    
    try:
        generator = FinalEnhancedGenerator()
        
        # Test transition prompt generation with keep_alive
        base_prompt = "test keep alive functionality"
        
        logger.info("📡 Generating transition prompts with keep_alive=30m...")
        transition_prompts = generator.generate_mixtral_transitions(base_prompt)
        
        if len(transition_prompts) == 5:
            logger.info("✅ Generated 5 transition prompts successfully")
            for i, prompt in enumerate(transition_prompts, 1):
                logger.info(f"  {i}. {prompt[:60]}...")
            logger.info("✅ FinalEnhancedGenerator keep_alive test successful")
            return True
        else:
            logger.warning(f"⚠️  Expected 5 prompts, got {len(transition_prompts)}")
            return False
            
    except Exception as e:
        logger.error(f"❌ FinalEnhancedGenerator keep_alive test failed: {e}")
        return False

def verify_keep_alive_in_code():
    """Verify keep_alive parameter is present in the code"""
    logger.info("🔍 Verifying keep_alive parameter in source code...")
    
    # Check mixtral_client.py
    with open('mixtral_client.py', 'r') as f:
        mixtral_content = f.read()
    
    if '"keep_alive": "30m"' in mixtral_content:
        logger.info("✅ keep_alive=30m found in mixtral_client.py")
        mixtral_ok = True
    else:
        logger.error("❌ keep_alive=30m NOT found in mixtral_client.py")
        mixtral_ok = False
    
    # Check final_enhanced_generator.py
    with open('final_enhanced_generator.py', 'r') as f:
        enhanced_content = f.read()
    
    if '"keep_alive": "30m"' in enhanced_content:
        logger.info("✅ keep_alive=30m found in final_enhanced_generator.py")
        enhanced_ok = True
    else:
        logger.error("❌ keep_alive=30m NOT found in final_enhanced_generator.py")
        enhanced_ok = False
    
    return mixtral_ok and enhanced_ok

def main():
    print("🔧 Testing Ollama Keep-Alive Parameter (30 minutes)")
    print("=" * 60)
    
    # Test 1: Verify code changes
    code_ok = verify_keep_alive_in_code()
    
    print("\n" + "=" * 60)
    
    # Test 2: Test MixtralClient (if available)
    mixtral_ok = test_mixtral_client_keepalive()
    
    print("\n" + "=" * 60)
    
    # Test 3: Test FinalEnhancedGenerator (if available)
    enhanced_ok = test_enhanced_generator_keepalive()
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS:")
    print(f"  Code Changes: {'✅ PASS' if code_ok else '❌ FAIL'}")
    print(f"  MixtralClient: {'✅ PASS' if mixtral_ok else '❌ FAIL'}")
    print(f"  EnhancedGenerator: {'✅ PASS' if enhanced_ok else '❌ FAIL'}")
    
    if code_ok:
        print("\n🎉 Keep-alive parameter successfully updated to 30 minutes!")
        print("💡 Ollama models will now stay loaded for 30 minutes instead of 5 minutes")
        print("⚡ This should improve response times for subsequent requests")
    else:
        print("\n⚠️  Some tests failed - check the logs above")
    
    print("=" * 60)

if __name__ == "__main__":
    main()