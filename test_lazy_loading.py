#!/usr/bin/env python3
"""
Test script to verify lazy loading implementation
"""
import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config.config import VideoSummarizationPipelineConfig
from src.pipeline.pipeline import VideoSummarizationPipeline


def test_lazy_loading():
    """Test that the lazy loading implementation works correctly"""
    print("Testing lazy loading implementation...")
    
    # Create a minimal test configuration
    config = VideoSummarizationPipelineConfig()
    config.exam_name = "lazy_loading_test"
    config.base_output_dir = "/tmp/lazy_loading_test"
    config.dataset_dirs = ["/tmp/test_dataset"]
    
    # Create test directories
    os.makedirs(config.base_output_dir, exist_ok=True)
    os.makedirs("/tmp/test_dataset", exist_ok=True)
    
    # Initialize pipeline
    print("1. Creating pipeline instance...")
    pipeline = VideoSummarizationPipeline(config)
    print("   ✓ Pipeline created successfully")
    
    # Test that check functions exist
    print("2. Testing check functions exist...")
    assert hasattr(pipeline, '_check_captions_exist'), "Missing _check_captions_exist method"
    assert hasattr(pipeline, '_check_visual_features_exist'), "Missing _check_visual_features_exist method"  
    assert hasattr(pipeline, '_check_text_features_exist'), "Missing _check_text_features_exist method"
    print("   ✓ All check functions exist")
    
    # Test check functions with empty dataset
    print("3. Testing check functions with empty dataset...")
    
    # Create a test dataset name  
    test_dataset_name = "test_dataset"
    
    # Create the dataset directory structure
    dataset_path = Path("/tmp/test_dataset")
    videos_dir = dataset_path / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    
    # Call check functions with dataset name
    captions_exist, missing_captions = pipeline._check_captions_exist(test_dataset_name)
    visual_exist, missing_visual = pipeline._check_visual_features_exist(test_dataset_name)  
    text_exist, missing_text = pipeline._check_text_features_exist(test_dataset_name)
    
    print(f"   Captions exist: {captions_exist}, missing: {missing_captions}")
    print(f"   Visual features exist: {visual_exist}, missing: {missing_visual}")
    print(f"   Text features exist: {text_exist}, missing: {missing_text}")
    print("   ✓ Check functions working")
    
    # Test Extractor lazy loading
    print("4. Testing Extractor lazy loading...")
    from src.config.config import ExtractorConfig
    extractor_config = ExtractorConfig()
    
    from src.llm.extractor import Extractor
    extractor = Extractor(extractor_config)
    
    # Verify model is not loaded initially
    assert not extractor._model_loaded, "Model should not be loaded initially"
    assert extractor.model is None, "Model should be None initially"
    assert extractor.processor is None, "Processor should be None initially"
    print("   ✓ Model not loaded initially")
    
    # Test that _load_model method exists
    assert hasattr(extractor, '_load_model'), "Missing _load_model method"
    print("   ✓ _load_model method exists")
    
    print("\n🎉 All lazy loading tests passed!")
    return True


def test_check_functions_with_data():
    """Test check functions with actual data structures"""
    print("\nTesting check functions with mock data...")
    
    # Create test directories with some mock files
    test_base = Path("/tmp/lazy_loading_test")
    test_dataset = test_base / "test_dataset" 
    
    # Create mock caption files
    captions_dir = test_dataset / "captions"
    captions_dir.mkdir(parents=True, exist_ok=True)
    (captions_dir / "video1.txt").write_text("Test caption")
    
    # Create mock visual features
    visual_dir = test_dataset / "visual_features"  
    visual_dir.mkdir(parents=True, exist_ok=True)
    (visual_dir / "video1.npy").write_text("mock npy")
    
    # Create mock text features
    text_dir = test_dataset / "text_features"
    text_dir.mkdir(parents=True, exist_ok=True)
    (text_dir / "video1_text.npy").write_text("mock npy")
    
    # Create mock videos directory
    videos_dir = test_dataset / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    (videos_dir / "video1.mp4").write_text("mock video")
    (videos_dir / "video2.mp4").write_text("mock video")
    
    # Test pipeline with mock data
    config = VideoSummarizationPipelineConfig()
    config.exam_name = "lazy_loading_test_data"
    config.base_output_dir = str(test_base)
    config.dataset_dirs = [str(test_dataset)]
    
    pipeline = VideoSummarizationPipeline(config)
    
    # Test check functions
    test_dataset_name = "test_dataset"
    captions_exist, missing_captions = pipeline._check_captions_exist(test_dataset_name)
    visual_exist, missing_visual = pipeline._check_visual_features_exist(test_dataset_name)
    text_exist, missing_text = pipeline._check_text_features_exist(test_dataset_name)
    
    print(f"   Captions exist: {captions_exist}, missing: {missing_captions}")
    print(f"   Visual features exist: {visual_exist}, missing: {missing_visual}")  
    print(f"   Text features exist: {text_exist}, missing: {missing_text}")
    
    # Should detect that some files are missing (video2 files)
    assert not captions_exist, "Should need captions for video2"
    assert not visual_exist, "Should need visual features for video2"
    assert not text_exist, "Should need text features for video2"
    
    print("   ✓ Check functions correctly detect missing files")
    print("🎉 Data check tests passed!")
    
    return True


if __name__ == "__main__":
    try:
        success = test_lazy_loading()
        if success:
            success = test_check_functions_with_data()
            
        if success:
            print("\n✅ All tests completed successfully!")
            print("Lazy loading implementation is working correctly.")
        else:
            print("\n❌ Some tests failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Cleanup
        import shutil
        if os.path.exists("/tmp/lazy_loading_test"):
            shutil.rmtree("/tmp/lazy_loading_test")
            print("\n🧹 Cleanup completed")
