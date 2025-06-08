#!/usr/bin/env python3
"""
Live ASR Demo Script

Interactive demo for easy_asr_server that allows:
- Press Enter to start recording
- Press Enter again to stop recording and get transcription
- Configurable pipeline and hotwords
- Press 'q' to quit

Requirements:
- sounddevice: pip install sounddevice
- numpy: pip install numpy
"""

import sys
import os
import time
import numpy as np
import sounddevice as sd
from typing import Optional

# Add the parent directory to the path to import easy_asr_server modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Use the new simplified API - much cleaner!
from easy_asr_server import EasyASR, get_available_pipelines, get_default_pipeline


class LiveASRDemo:
    """Interactive ASR demo with live recording capabilities."""
    
    def __init__(self, pipeline: str = None, device: str = "auto", hotwords: str = ""):
        """
        Initialize the Live ASR Demo.
        
        Args:
            pipeline: ASR pipeline type ('sensevoice' or 'paraformer')
            device: Device for inference ('auto', 'cpu', 'cuda', etc.)
            hotwords: Space-separated hotwords string
        """
        self.pipeline = pipeline or get_default_pipeline()
        self.device = device
        self.hotwords = hotwords
        self.sample_rate = 16000
        self.channels = 1
        
        # Recording state
        self.is_recording = False
        self.audio_data = []
        self.stream = None
        
        # ASR engine - using the new simplified API
        self.asr_engine = None
        
        print(f"🎤 Live ASR Demo")
        print(f"Pipeline: {self.pipeline}")
        print(f"Device: {self.device}")
        print(f"Hotwords: '{self.hotwords}' (empty if none)")
        print("-" * 50)
    
    def initialize_asr(self):
        """Initialize the ASR engine using the new EasyASR API."""
        try:
            print("🔄 Initializing ASR components...")
            
            # This is much simpler now - just one line!
            self.asr_engine = EasyASR(
                pipeline=self.pipeline,
                device=self.device,
                hotwords=self.hotwords,
                log_level="INFO",  # Show some progress info
                auto_init=True     # Initialize automatically
            )
            
            # Get info about the initialized engine
            info = self.asr_engine.get_info()
            print(f"📱 Using device: {info['resolved_device']}")
            print(f"📦 Pipeline loaded: {info['pipeline']}")
            
            if self.asr_engine.is_healthy():
                print("✅ ASR engine is healthy and ready!")
                return True
            else:
                print("❌ ASR engine health check failed!")
                return False
                
        except Exception as e:
            print(f"❌ Failed to initialize ASR: {e}")
            return False
    
    def audio_callback(self, indata, frames, time, status):
        """Callback function for audio recording."""
        if status:
            print(f"⚠️  Audio status: {status}")
        
        if self.is_recording:
            # Convert to mono if needed and append to buffer
            if indata.shape[1] > 1:
                mono_data = np.mean(indata, axis=1)
            else:
                mono_data = indata[:, 0]
            self.audio_data.append(mono_data.copy())
    
    def start_recording(self):
        """Start audio recording."""
        if self.is_recording:
            print("⚠️  Already recording!")
            return
        
        self.is_recording = True
        self.audio_data = []
        
        try:
            # Start audio stream
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=self.audio_callback,
                dtype=np.float32
            )
            self.stream.start()
            print("🔴 Recording started... Press Enter to stop")
            
        except Exception as e:
            print(f"❌ Failed to start recording: {e}")
            self.is_recording = False
    
    def stop_recording(self):
        """Stop audio recording and return recorded audio."""
        if not self.is_recording:
            print("⚠️  Not recording!")
            return None
        
        self.is_recording = False
        
        try:
            # Stop and close audio stream
            self.stream.stop()
            self.stream.close()
            
            if not self.audio_data:
                print("❌ No audio data recorded!")
                return None
            
            # Concatenate all audio chunks
            audio_array = np.concatenate(self.audio_data, axis=0)
            duration = len(audio_array) / self.sample_rate
            
            print(f"⏹️  Recording stopped. Duration: {duration:.2f}s, Samples: {len(audio_array)}")
            
            # Ensure minimum length
            if len(audio_array) < 512:
                print("⚠️  Recording too short (minimum 512 samples required)")
                return None
            
            return audio_array.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Failed to stop recording: {e}")
            return None
    
    def recognize_audio(self, audio_array: np.ndarray) -> Optional[str]:
        """Recognize speech from audio array."""
        if self.asr_engine is None:
            print("❌ ASR engine not initialized!")
            return None
        
        try:
            print("🤔 Recognizing speech...")
            start_time = time.time()
            
            # Recognition is now much simpler too!
            result = self.asr_engine.recognize(audio_array)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            print(f"⚡ Recognition completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            print(f"❌ Recognition failed: {e}")
            return None
    
    def run(self):
        """Main demo loop."""
        # Initialize ASR
        if not self.initialize_asr():
            print("❌ Failed to initialize ASR. Exiting...")
            return
        
        print("\n📋 Instructions:")
        print("• Press Enter to start recording")
        print("• Press Enter again to stop recording and get transcription")
        print("• Type 'q' and press Enter to quit")
        print("• Type 'h' and press Enter to show hotwords")
        print("• Type 'i' and press Enter to show engine info")
        print("-" * 50)
        
        recording_state = False  # False = ready to record, True = currently recording
        
        while True:
            try:
                if recording_state:
                    user_input = input("🔴 [Recording] Press Enter to stop: ").strip()
                else:
                    user_input = input("⚪ [Ready] Press Enter to record (q=quit, h=hotwords, i=info): ").strip()
                
                # Handle commands
                if user_input.lower() == 'q':
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'h':
                    if self.hotwords:
                        print(f"🔤 Current hotwords: '{self.hotwords}'")
                    else:
                        print("🔤 No hotwords configured")
                    continue
                elif user_input.lower() == 'i':
                    # Show engine info using the new API
                    info = self.asr_engine.get_info()
                    print("ℹ️  Engine Information:")
                    for key, value in info.items():
                        print(f"   {key}: {value}")
                    continue
                elif user_input == "":
                    # Empty input (Enter pressed)
                    if not recording_state:
                        # Start recording
                        self.start_recording()
                        recording_state = True
                    else:
                        # Stop recording and recognize
                        audio_array = self.stop_recording()
                        recording_state = False
                        
                        if audio_array is not None:
                            result = self.recognize_audio(audio_array)
                            if result:
                                print(f"📝 Result: {result}")
                            else:
                                print("❌ No transcription result")
                        print("-" * 50)
                else:
                    print("⚠️  Unknown command. Use Enter to record, 'q' to quit, 'h' for hotwords, 'i' for info.")
                    
            except KeyboardInterrupt:
                print("\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                break
        
        # Cleanup
        if self.is_recording:
            self.stop_recording()


def main():
    """Main entry point with command line argument parsing."""
    import argparse
    
    # Get available pipelines dynamically
    available_pipelines = list(get_available_pipelines().keys())
    default_pipeline = get_default_pipeline()
    
    parser = argparse.ArgumentParser(description="Live ASR Demo for easy_asr_server")
    parser.add_argument(
        "--pipeline", "-p",
        default=default_pipeline,
        choices=available_pipelines,
        help=f"ASR pipeline type (default: {default_pipeline})"
    )
    parser.add_argument(
        "--device", "-d",
        default="auto",
        help="Device for inference (default: auto)"
    )
    parser.add_argument(
        "--hotwords", "-w",
        default="",
        help="Space-separated hotwords string (default: none)"
    )
    
    args = parser.parse_args()
    
    # Check if sounddevice is available
    try:
        import sounddevice as sd
    except ImportError:
        print("❌ Error: sounddevice library not found!")
        print("Please install it with: pip install sounddevice")
        sys.exit(1)
    
    # Create and run demo
    demo = LiveASRDemo(
        pipeline=args.pipeline,
        device=args.device,
        hotwords=args.hotwords
    )
    
    demo.run()


if __name__ == "__main__":
    main() 