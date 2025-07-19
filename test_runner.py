#!/usr/bin/env python3
"""
Test Runner for AI Video Generation Pipeline
Runs comprehensive tests for all components before starting the main application
"""

import sys
import os
import unittest
import logging
import traceback
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure test logging
logging.basicConfig(
    level=logging.INFO,
    format='[TEST] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

class TestRunner:
    """Main test runner that orchestrates all tests"""
    
    def __init__(self):
        self.test_results = {
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'skipped': 0
        }
        self.failed_tests = []
        
    def run_all_tests(self):
        """Run all test suites"""
        logging.info("🧪 Starting AI Video Generation Pipeline Tests")
        logging.info("=" * 60)
        
        # Test suites to run
        test_suites = [
            'tests.test_imports',
            'tests.test_models', 
            'tests.test_extensions',
            'tests.test_directories',
            'tests.test_pipeline',
            'tests.test_ui_components'
        ]
        
        overall_success = True
        
        for suite_name in test_suites:
            logging.info(f"\n📋 Running test suite: {suite_name}")
            logging.info("-" * 40)
            
            try:
                # Import and run test suite
                suite_module = __import__(suite_name, fromlist=[''])
                loader = unittest.TestLoader()
                suite = loader.loadTestsFromModule(suite_module)
                
                # Run tests
                runner = unittest.TextTestRunner(
                    verbosity=2,
                    stream=sys.stdout,
                    buffer=True
                )
                result = runner.run(suite)
                
                # Track results
                self.test_results['passed'] += result.testsRun - len(result.failures) - len(result.errors)
                self.test_results['failed'] += len(result.failures)
                self.test_results['errors'] += len(result.errors)
                self.test_results['skipped'] += len(result.skipped)
                
                # Track failed tests
                for failure in result.failures:
                    self.failed_tests.append(f"{suite_name}: {failure[0]}")
                for error in result.errors:
                    self.failed_tests.append(f"{suite_name}: {error[0]}")
                
                if result.failures or result.errors:
                    overall_success = False
                    logging.error(f"❌ {suite_name} had {len(result.failures)} failures and {len(result.errors)} errors")
                else:
                    logging.info(f"✅ {suite_name} passed all tests")
                    
            except ImportError as e:
                logging.error(f"❌ Could not import test suite {suite_name}: {e}")
                overall_success = False
                self.test_results['errors'] += 1
            except Exception as e:
                logging.error(f"❌ Error running test suite {suite_name}: {e}")
                logging.error(traceback.format_exc())
                overall_success = False
                self.test_results['errors'] += 1
        
        # Print summary
        self.print_summary(overall_success)
        return overall_success
    
    def print_summary(self, overall_success):
        """Print test summary"""
        logging.info("\n" + "=" * 60)
        logging.info("🧪 TEST SUMMARY")
        logging.info("=" * 60)
        
        total_tests = sum(self.test_results.values())
        logging.info(f"Total Tests: {total_tests}")
        logging.info(f"✅ Passed: {self.test_results['passed']}")
        logging.info(f"❌ Failed: {self.test_results['failed']}")
        logging.info(f"💥 Errors: {self.test_results['errors']}")
        logging.info(f"⏭️  Skipped: {self.test_results['skipped']}")
        
        if self.failed_tests:
            logging.info(f"\n❌ Failed Tests:")
            for test in self.failed_tests:
                logging.info(f"  - {test}")
        
        if overall_success:
            logging.info(f"\n🎉 ALL TESTS PASSED! System is ready to run.")
        else:
            logging.error(f"\n💥 SOME TESTS FAILED! Please fix issues before running the application.")
        
        logging.info("=" * 60)

def main():
    """Main test runner entry point"""
    runner = TestRunner()
    success = runner.run_all_tests()
    
    if not success:
        sys.exit(1)  # Exit with error code if tests failed
    
    return success

if __name__ == "__main__":
    main()