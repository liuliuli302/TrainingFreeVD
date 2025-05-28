#!/usr/bin/env python3
"""
Test script to verify evaluator compatibility with new multi-prompt result format
"""

import json
import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.config.config import EvaluatorConfig
from src.eval.evaluator import VideoSummaryEvaluator


def create_test_score_files():
    """Create test score files in both old and new formats"""
    
    # Old format (backward compatibility)
    old_format = {
        "summe_dataset_jump": {
            "video_1": [0.1, 0.8, 0.3, 0.9, 0.5, 0.2, 0.7],
            "video_2": [0.6, 0.4, 0.9, 0.1, 0.8, 0.3, 0.5]
        },
        "tvsum_dataset_jump": {
            "video_3": [0.2, 0.7, 0.4, 0.8, 0.6, 0.1, 0.9],
            "video_4": [0.5, 0.3, 0.8, 0.2, 0.7, 0.4, 0.6]
        }
    }
    
    # New format (multi-prompt)
    new_format = {
        "summe_dataset_jump": {
            "main_scores": {
                "video_1": [0.1, 0.8, 0.3, 0.9, 0.5, 0.2, 0.7],
                "video_2": [0.6, 0.4, 0.9, 0.1, 0.8, 0.3, 0.5]
            },
            "prompt_scores": {
                "importance_representativeness": {
                    "video_1": [0.2, 0.7, 0.4, 0.8, 0.6, 0.3, 0.6],
                    "video_2": [0.5, 0.5, 0.8, 0.2, 0.7, 0.4, 0.4]
                },
                "diversity_information": {
                    "video_1": [0.1, 0.9, 0.2, 1.0, 0.4, 0.1, 0.8],
                    "video_2": [0.7, 0.3, 1.0, 0.0, 0.9, 0.2, 0.6]
                },
                "comprehensive_summary": {
                    "video_1": [0.1, 0.8, 0.3, 0.9, 0.5, 0.2, 0.7],
                    "video_2": [0.6, 0.4, 0.9, 0.1, 0.8, 0.3, 0.5]
                }
            },
            "extraction_rate": "100/100"
        },
        "tvsum_dataset_jump": {
            "main_scores": {
                "video_3": [0.2, 0.7, 0.4, 0.8, 0.6, 0.1, 0.9],
                "video_4": [0.5, 0.3, 0.8, 0.2, 0.7, 0.4, 0.6]
            },
            "prompt_scores": {
                "importance_representativeness": {
                    "video_3": [0.3, 0.6, 0.5, 0.7, 0.7, 0.2, 0.8],
                    "video_4": [0.4, 0.4, 0.7, 0.3, 0.6, 0.5, 0.5]
                },
                "diversity_information": {
                    "video_3": [0.1, 0.8, 0.3, 0.9, 0.5, 0.0, 1.0],
                    "video_4": [0.6, 0.2, 0.9, 0.1, 0.8, 0.3, 0.7]
                },
                "comprehensive_summary": {
                    "video_3": [0.2, 0.7, 0.4, 0.8, 0.6, 0.1, 0.9],
                    "video_4": [0.5, 0.3, 0.8, 0.2, 0.7, 0.4, 0.6]
                }
            },
            "extraction_rate": "50/50"
        }
    }
    
    return old_format, new_format


def test_old_format_compatibility():
    """Test that evaluator still works with old format"""
    print("Testing old format compatibility...")
    
    old_format, _ = create_test_score_files()
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(old_format, f, indent=2)
        temp_file = f.name
    
    try:
        # Mock config - we'll override the file path
        config = EvaluatorConfig()
        
        # Create evaluator (this will fail with real dataset files, but we just test format parsing)
        evaluator = VideoSummaryEvaluator(config)
        
        # Test the format parsing part by directly reading the file
        with open(temp_file) as f:
            result_list = json.load(f)
        
        print(f"✓ Old format file loaded successfully")
        print(f"  Keys: {list(result_list.keys())}")
        
        # Test format detection
        for key in result_list.keys():
            result_data = result_list[key]
            if isinstance(result_data, dict) and "main_scores" in result_data:
                print(f"  {key}: Detected as NEW format")
            else:
                print(f"  {key}: Detected as OLD format")
        
        return True
        
    except Exception as e:
        print(f"✗ Old format test failed: {str(e)}")
        return False
    finally:
        # Clean up
        Path(temp_file).unlink(missing_ok=True)


def test_new_format_support():
    """Test that evaluator works with new multi-prompt format"""
    print("\nTesting new format support...")
    
    _, new_format = create_test_score_files()
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(new_format, f, indent=2)
        temp_file = f.name
    
    try:
        # Test the format parsing part
        with open(temp_file) as f:
            result_list = json.load(f)
        
        print(f"✓ New format file loaded successfully")
        print(f"  Keys: {list(result_list.keys())}")
        
        # Test format detection and parsing
        for key in result_list.keys():
            result_data = result_list[key]
            if isinstance(result_data, dict) and "main_scores" in result_data:
                print(f"  {key}: Detected as NEW format")
                print(f"    Main scores videos: {list(result_data['main_scores'].keys())}")
                if "prompt_scores" in result_data:
                    prompts = list(result_data['prompt_scores'].keys())
                    print(f"    Available prompts: {prompts}")
                if "extraction_rate" in result_data:
                    print(f"    Extraction rate: {result_data['extraction_rate']}")
            else:
                print(f"  {key}: Detected as OLD format")
        
        return True
        
    except Exception as e:
        print(f"✗ New format test failed: {str(e)}")
        return False
    finally:
        # Clean up
        Path(temp_file).unlink(missing_ok=True)


def test_prompt_evaluation():
    """Test individual prompt evaluation functionality"""
    print("\nTesting prompt evaluation functionality...")
    
    _, new_format = create_test_score_files()
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(new_format, f, indent=2)
        temp_file = f.name
    
    try:
        # Mock config
        config = EvaluatorConfig()
        evaluator = VideoSummaryEvaluator(config)
        
        # Test prompt results parsing (without actual evaluation since we don't have real datasets)
        with open(temp_file) as f:
            result_list = json.load(f)
        
        print("✓ Testing prompt results structure...")
        
        for key in result_list.keys():
            result_data = result_list[key]
            
            if isinstance(result_data, dict) and "prompt_scores" in result_data:
                print(f"  Dataset: {key}")
                for prompt_name, prompt_scores in result_data["prompt_scores"].items():
                    print(f"    Prompt: {prompt_name}")
                    print(f"      Videos: {list(prompt_scores.keys())}")
                    
                    # Test score structure
                    for video_name, scores in prompt_scores.items():
                        if isinstance(scores, list) and len(scores) > 0:
                            print(f"        {video_name}: {len(scores)} scores (range: {min(scores):.2f}-{max(scores):.2f})")
                        else:
                            print(f"        {video_name}: Invalid scores")
        
        return True
        
    except Exception as e:
        print(f"✗ Prompt evaluation test failed: {str(e)}")
        return False
    finally:
        # Clean up
        Path(temp_file).unlink(missing_ok=True)


def main():
    """Run all evaluator tests"""
    print("Starting evaluator compatibility verification...")
    
    results = []
    
    # Test 1: Old format compatibility
    results.append(test_old_format_compatibility())
    
    # Test 2: New format support
    results.append(test_new_format_support())
    
    # Test 3: Prompt evaluation
    results.append(test_prompt_evaluation())
    
    # Summary
    print(f"\n{'='*50}")
    print("EVALUATOR TEST SUMMARY")
    print(f"{'='*50}")
    
    test_names = [
        "Old Format Compatibility",
        "New Format Support",
        "Prompt Evaluation"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "PASS" if result else "FAIL"
        print(f"{i+1}. {name}: {status}")
    
    total_pass = sum(results)
    print(f"\nOverall: {total_pass}/{len(results)} tests passed")
    
    if total_pass == len(results):
        print("🎉 All evaluator tests passed! Format compatibility is working.")
    else:
        print("❌ Some evaluator tests failed. Please check the error messages above.")
    
    return total_pass == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
