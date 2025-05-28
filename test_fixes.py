#!/usr/bin/env python3
"""
Test script to verify meta tensor fixes and multi-prompt functionality
"""

import os
import sys
import torch
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.config.config import LLMHandlerConfig, LLMQueryConfig
from src.llm.llm_handler import LLMHandler
from src.llm.llm_query import LLMQuery


def test_meta_tensor_fix():
    """Test if meta tensor issues are resolved"""
    print("Testing meta tensor fix...")
    
    try:
        # Test with problematic device_map="auto" 
        config = LLMHandlerConfig(
            model_type="local",
            pretrained="lmms-lab/LLaVA-Video-7B-Qwen2",
            device_map="auto"  # This might cause meta tensor issues
        )
        
        print("Creating LLMHandler with device_map='auto'...")
        handler = LLMHandler(config)
        print("✓ LLMHandler created successfully with auto device mapping")
        
        # Clean up
        del handler
        torch.cuda.empty_cache()
        
        # Test with fixed device_map
        config.device_map = "cuda:0"
        print("Creating LLMHandler with device_map='cuda:0'...")
        handler = LLMHandler(config)
        print("✓ LLMHandler created successfully with fixed device mapping")
        
        # Clean up
        del handler
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"✗ Meta tensor test failed: {error_msg}")
        
        if "meta" in error_msg.lower() or "cannot copy" in error_msg.lower():
            print("Meta tensor error detected - this is expected if the fix isn't working")
        else:
            print("Different error occurred")
            
        traceback.print_exc()
        return False


def test_multi_prompt_system():
    """Test the new multi-prompt system"""
    print("\nTesting multi-prompt system...")
    
    try:
        # Create test config
        llm_handler_config = LLMHandlerConfig(
            model_type="local",
            pretrained="lmms-lab/LLaVA-Video-7B-Qwen2",
            device_map="cuda:0"
        )
        
        llm_query_config = LLMQueryConfig(
            dataset_dir="/path/to/test",
            result_dir="/path/to/results",
            scores_dir="/path/to/scores",
            frame_interval=15
        )
        
        # Create handlers
        llm_handler = LLMHandler(llm_handler_config)
        llm_query = LLMQuery(llm_query_config, llm_handler)
        
        # Check that video_summary_prompts exist
        print(f"Number of video summary prompts: {len(llm_query.video_summary_prompts)}")
        
        for i, prompt_info in enumerate(llm_query.video_summary_prompts):
            print(f"  {i+1}. {prompt_info['name']}")
        
        # Test score extraction patterns
        test_responses = [
            "[0.1, 0.8, 0.3, 0.9, 0.5]",
            "Scores: [0.2, 0.7, 0.4]",
            "0.5 0.8 0.2 0.9",
            "Score: 0.6\nScore: 0.8\nScore: 0.3",
        ]
        
        print("\nTesting score extraction:")
        for response in test_responses:
            scores = llm_query.extract_scores_from_response(response)
            print(f"  Input: {response[:30]}...")
            print(f"  Extracted: {scores}")
        
        print("✓ Multi-prompt system test completed successfully")
        
        # Clean up
        del llm_handler
        del llm_query
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"✗ Multi-prompt test failed: {str(e)}")
        traceback.print_exc()
        return False


def test_config_compatibility():
    """Test that config changes maintain backward compatibility"""
    print("\nTesting config compatibility...")
    
    try:
        # Test old-style config creation
        config = LLMQueryConfig(
            dataset_dir="/test/dataset",
            result_dir="/test/result",
            scores_dir="/test/scores",
            frame_interval=15
        )
        
        print(f"✓ Config created with default prompts")
        print(f"  Score pattern: {config.score_pattern}")
        print(f"  Dataset files: {len(config.dataset_file_names)}")
        print(f"  HDF5 files: {config.hdf5_file_names}")
        
        # Test with custom prompts (backward compatibility)
        config_custom = LLMQueryConfig(
            dataset_dir="/test/dataset",
            result_dir="/test/result", 
            scores_dir="/test/scores",
            frame_interval=15,
            first_prompt="Custom first prompt",
            second_prompt="Custom second prompt"
        )
        
        print(f"✓ Config created with custom prompts")
        print(f"  First prompt: {config_custom.first_prompt[:50]}...")
        print(f"  Second prompt: {config_custom.second_prompt[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Config compatibility test failed: {str(e)}")
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("Starting meta tensor fix and multi-prompt system verification...")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available - some tests may fail")
    else:
        print(f"✓ CUDA available with {torch.cuda.device_count()} devices")
    
    results = []
    
    # Test 1: Meta tensor fix
    results.append(test_meta_tensor_fix())
    
    # Test 2: Multi-prompt system  
    results.append(test_multi_prompt_system())
    
    # Test 3: Config compatibility
    results.append(test_config_compatibility())
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    
    test_names = [
        "Meta Tensor Fix",
        "Multi-Prompt System", 
        "Config Compatibility"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "PASS" if result else "FAIL"
        print(f"{i+1}. {name}: {status}")
    
    total_pass = sum(results)
    print(f"\nOverall: {total_pass}/{len(results)} tests passed")
    
    if total_pass == len(results):
        print("🎉 All tests passed! The fixes appear to be working correctly.")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
    
    return total_pass == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
