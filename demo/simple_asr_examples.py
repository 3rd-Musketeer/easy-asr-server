#!/usr/bin/env python3
"""
Simple ASR Examples

This script demonstrates various ways to use the new EasyASR API for speech recognition.
It shows both simple one-shot usage and more advanced patterns.
"""

import sys
import os

# Add the parent directory to the path to import easy_asr_server modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the new simplified API
from easy_asr_server import (
    EasyASR, 
    create_asr_engine, 
    recognize,
    get_available_pipelines,
    get_default_pipeline
)


def example_1_simple_usage():
    """Example 1: Simplest possible usage"""
    print("=" * 60)
    print("Example 1: Simple Usage")
    print("=" * 60)
    
    # Check if we have a sample audio file
    sample_file = "../sample.wav"
    if not os.path.exists(sample_file):
        print("⚠️  Sample audio file not found. Skipping this example.")
        return
    
    try:
        # The simplest way - just one line!
        result = recognize(sample_file)
        print(f"📝 Recognition result: {result}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def example_2_with_configuration():
    """Example 2: Usage with custom configuration"""
    print("\n" + "=" * 60)
    print("Example 2: Custom Configuration")
    print("=" * 60)
    
    sample_file = "../sample.wav"
    if not os.path.exists(sample_file):
        print("⚠️  Sample audio file not found. Skipping this example.")
        return
    
    try:
        # With custom pipeline and hotwords
        result = recognize(
            audio_input=sample_file,
            pipeline="sensevoice",
            device="cpu",  # Force CPU for this example
            hotwords="你好 世界 语音识别"
        )
        print(f"📝 Recognition result with hotwords: {result}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def example_3_reusable_engine():
    """Example 3: Creating a reusable engine for multiple recognitions"""
    print("\n" + "=" * 60)
    print("Example 3: Reusable Engine")
    print("=" * 60)
    
    sample_file = "../sample.wav"
    if not os.path.exists(sample_file):
        print("⚠️  Sample audio file not found. Skipping this example.")
        return
    
    try:
        # Create an engine once, use it multiple times
        print("🔄 Creating ASR engine...")
        asr = create_asr_engine(
            pipeline="sensevoice",
            device="auto",
            hotwords="语音 识别 测试",
            log_level="INFO"
        )
        
        # Show engine info
        info = asr.get_info()
        print("ℹ️  Engine info:")
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        # Use it multiple times (simulated)
        print("\n🎤 Performing recognition...")
        result = asr.recognize(sample_file)
        print(f"📝 First result: {result}")
        
        # You could use it again with different hotwords
        result2 = asr.recognize(sample_file, hotwords="另一组 热词")
        print(f"📝 Second result (different hotwords): {result2}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def example_4_context_manager():
    """Example 4: Using context manager for automatic cleanup"""
    print("\n" + "=" * 60)
    print("Example 4: Context Manager")
    print("=" * 60)
    
    sample_file = "../sample.wav"
    if not os.path.exists(sample_file):
        print("⚠️  Sample audio file not found. Skipping this example.")
        return
    
    try:
        # Using with statement for automatic cleanup
        with EasyASR(pipeline="sensevoice", device="auto") as asr:
            print(f"🔋 Engine healthy: {asr.is_healthy()}")
            result = asr.recognize(sample_file)
            print(f"📝 Result: {result}")
        # Engine is automatically cleaned up here
        print("✅ Engine cleaned up automatically")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def example_5_advanced_init():
    """Example 5: Advanced initialization patterns"""
    print("\n" + "=" * 60)
    print("Example 5: Advanced Initialization")
    print("=" * 60)
    
    try:
        # Create engine without auto-initialization
        asr = EasyASR(
            pipeline="sensevoice",
            device="auto",
            auto_init=False  # Don't initialize yet
        )
        
        print("🔄 Engine created but not initialized")
        print(f"🔋 Is healthy: {asr.is_healthy()}")  # Should be False
        
        # Initialize manually when ready
        print("🔄 Initializing manually...")
        success = asr.initialize()
        print(f"✅ Initialization successful: {success}")
        print(f"🔋 Is healthy now: {asr.is_healthy()}")  # Should be True
        
        # Show detailed info
        info = asr.get_info()
        print("ℹ️  Detailed engine information:")
        for key, value in info.items():
            print(f"   {key}: {value}")
            
    except Exception as e:
        print(f"❌ Error: {e}")


def show_available_options():
    """Show available pipelines and configurations"""
    print("=" * 60)
    print("Available Options")
    print("=" * 60)
    
    print(f"🔧 Default pipeline: {get_default_pipeline()}")
    
    print("\n🔧 Available pipelines:")
    pipelines = get_available_pipelines()
    for name, config in pipelines.items():
        print(f"   • {name}")
        if 'components' in config:
            print(f"     Components: {list(config['components'].keys())}")


def main():
    """Run all examples"""
    print("🎤 EasyASR API Examples")
    print("This script demonstrates various usage patterns of the new simplified API.")
    
    # Show available options first
    show_available_options()
    
    # Run examples
    example_1_simple_usage()
    example_2_with_configuration()
    example_3_reusable_engine()
    example_4_context_manager()
    example_5_advanced_init()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print("\n📖 Key takeaways:")
    print("• Use recognize() for simple one-shot recognition")
    print("• Use create_asr_engine() or EasyASR() for reusable engines")
    print("• Use context managers (with statement) for automatic cleanup")
    print("• The API handles all the complex initialization internally")
    print("• You can configure pipeline, device, and hotwords easily")


if __name__ == "__main__":
    main() 