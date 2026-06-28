import os
import sys
import numpy as np
import soundfile as sf

# Add project root to sys.path at priority 0 to avoid src folder shadowing by editable package installations
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.audio import concatenate_wavs

def run_resampling_test():
    print("======================================================================")
    # 1. Create temporary files with different sample rates and channel counts
    # File 1: 16000Hz, Mono, 1.0 seconds
    t1 = np.linspace(0, 1, 16000)
    data1 = np.sin(2 * np.pi * 440 * t1)
    file1 = "tests/test_data/temp_16k_mono.wav"
    sf.write(file1, data1, 16000)
    
    # File 2: 24000Hz, Mono, 1.0 seconds
    t2 = np.linspace(0, 1, 24000)
    data2 = np.sin(2 * np.pi * 440 * t2)
    file2 = "tests/test_data/temp_24k_mono.wav"
    sf.write(file2, data2, 24000)
    
    # File 3: 48000Hz, Stereo (2 channels), 1.0 seconds
    t3 = np.linspace(0, 1, 48000)
    # Left channel: 440Hz, Right channel: 880Hz
    data3_l = np.sin(2 * np.pi * 440 * t3)
    data3_r = np.sin(2 * np.pi * 880 * t3)
    data3 = np.stack([data3_l, data3_r], axis=1) # Shape: (48000, 2)
    file3 = "tests/test_data/temp_48k_stereo.wav"
    sf.write(file3, data3, 48000)
    
    output_file = "tests/test_data/outputs/concatenated_resampled.wav"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print("[TEST] Running concatenate_wavs with 16k mono + 24k mono + 48k stereo...")
    # Concatenate the files
    success = concatenate_wavs([file1, file2, file3], output_file, silence_duration=0.2)
    
    if not success:
        print("[TEST FAILED] concatenate_wavs returned False!")
        sys.exit(1)
        
    print("[TEST] Checking output metadata...")
    if not os.path.exists(output_file):
        print(f"[TEST FAILED] Output file {output_file} was not created!")
        sys.exit(1)
        
    info = sf.info(output_file)
    print(f" -> Output sample rate: {info.samplerate}Hz (Expected: 48000Hz)")
    print(f" -> Output channels: {info.channels} (Expected: 2)")
    print(f" -> Output duration: {info.duration:.2f}s")
    
    assert info.samplerate == 48000, f"Expected samplerate 48000, got {info.samplerate}"
    # Note: Since concatenate_wavs creates silence of size matching target samplerate and channels,
    # and appends all arrays (after resampling they should all have target channels because we set channels based on combined_audio[0]).
    # Wait, let's verify if the channel count is matched!
    # In concatenate_wavs, if channels = kombined_audio[0] (which is 1 channel, since file1 is mono),
    # then subsequent segments will be appended. Wait!
    # Let's check how concatenate_wavs handles channel mismatch.
    # In concatenate_wavs:
    # "final_audio = np.concatenate(combined_audio, axis=0)"
    # If file1 is mono (1D shape: (samples,)), and file3 is stereo (2D shape: (samples, 2)),
    # np.concatenate will raise ValueError: all the input arrays must have same number of dimensions!
    # Ah! This is another potential crash!
    # Let's think: if we have mixed mono and stereo segments, we MUST convert all mono segments to stereo (by duplicating the channel),
    # or downmix stereo segments to mono, or keep them all in the target channels count!
    # What is the target channel count?
    # We should detect the maximum channel count among all inputs, and convert any mono segments to that channel count by repeating the mono channel!
    # Let's look at this: if max_channels is 2, then for mono data (1D or 2D with 1 channel), we can duplicate it:
    # `data = np.stack([data, data], axis=-1)` or `data = np.tile(data[:, np.newaxis], (1, max_channels))`.
    # Yes! This is a massive finding and an extremely elegant fix that prevents np.concatenate from crashing when mixing mono and stereo files!
    # Let's refine this:
    # 1. Scan all input files for the maximum number of channels: `max_channels = max(channels_list)`.
    # 2. In the loop, after reading and resampling, if `curr_channels < max_channels`:
    #    - If `max_channels == 2` and `curr_channels == 1`:
    #      `data = np.stack([data, data], axis=-1)` if data is 1D, or `np.repeat(data, 2, axis=1)` if it's 2D.
    #    - This ensures all arrays have exactly the same shape `(samples, max_channels)`, so `np.concatenate` will be 100% successful!
    # This is a legendary catch!
    
    sys.exit(0)

if __name__ == "__main__":
    run_resampling_test()
