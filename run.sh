#!/bin/bash
# Video Generation Environment Launcher with Comprehensive Testing

echo "🎬 AI Video Generation Pipeline"
echo "=============================="

# Function to run tests
run_tests() {
    echo ""
    echo "🧪 Running Comprehensive Tests..."
    echo "================================="
    
    # Run test suite
    python test_runner.py
    
    # Check test exit code
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ All tests passed! System is ready to run."
        echo ""
        return 0
    else
        echo ""
        echo "❌ Some tests failed! Please check the output above."
        echo "🔧 Fix the issues before running the application."
        echo ""
        read -p "Do you want to continue anyway? (y/N): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "⚠️  Continuing with potential issues..."
            return 0
        else
            echo "🛑 Stopping execution. Please fix the issues and try again."
            exit 1
        fi
    fi
}

# Parse command line arguments
RUN_TESTS=true
FORCE_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-tests)
            RUN_TESTS=false
            shift
            ;;
        --force)
            FORCE_RUN=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-tests    Skip the test suite and run directly"
            echo "  --force         Force run even if tests fail (no prompt)"
            echo "  --help, -h      Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Run with tests (default)"
            echo "  $0 --skip-tests      # Skip tests and run directly"
            echo "  $0 --force           # Run tests but continue even if they fail"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Activate conda environment
echo "Activating conda environment: video-generation..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate video-generation

# Verify environment
echo "Current environment: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"

# Run tests if requested
if [ "$RUN_TESTS" = true ]; then
    if [ "$FORCE_RUN" = true ]; then
        echo ""
        echo "🧪 Running tests in force mode..."
        python test_runner.py
        echo "⚠️  Continuing regardless of test results (force mode)..."
    else
        run_tests
    fi
else
    echo ""
    echo "⏭️  Skipping tests as requested..."
fi

# Pre-flight checks
echo ""
echo "🔍 Pre-flight Checks..."
echo "======================"

# Check if required directories exist
if [ ! -d "output" ]; then
    echo "📁 Creating output directory..."
    mkdir -p output/frames output/videos
fi

if [ ! -d "in/individual" ]; then
    echo "📁 Creating face input directory..."
    mkdir -p in/individual
    echo "⚠️  No face images found. Roop functionality will be limited."
    echo "   Add face images to in/individual/{person_name}/ for face swapping."
fi

# Check for face images
face_count=$(find in/individual -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" 2>/dev/null | wc -l)
if [ $face_count -gt 0 ]; then
    echo "👤 Found $face_count face images for Roop face swapping"
else
    echo "⚠️  No face images found - Roop face swapping will be disabled"
fi

# Launch application
echo ""
echo "🚀 Starting Application..."
echo "========================="
echo "📱 Open http://localhost:8003 in your browser"
echo "🛑 Press Ctrl+C to stop the application"
echo ""

python app.py