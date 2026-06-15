import os
import soundfile as sf
import numpy as np

def concatenate_wavs(input_paths: list[str], output_path: str, silence_duration: float = 0.3) -> bool:
    """Concatenates multiple WAV files into a single WAV file with a short silence gap.
    
    Also normalizes the combined amplitude to prevent clipping and cleans up the input files.
    
    Args:
        input_paths: List of paths to temporary WAV files.
        output_path: Path to write the final concatenated WAV file.
        silence_duration: Duration of silence gap in seconds.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    if not input_paths:
        return False
        
    combined_audio = []
    samplerate = None
    channels = None
    
    for path in input_paths:
        if not os.path.exists(path):
            continue
            
        try:
            data, sr = sf.read(path)
        except Exception as e:
            print(f"Error reading WAV file {path}: {e}")
            continue
            
        # Initialize configuration from the first valid file
        if samplerate is None:
            samplerate = sr
            if len(data.shape) > 1:
                channels = data.shape[1]
            else:
                channels = 1
        elif sr != samplerate:
            # Standard engines have uniform sample rates, but check just in case
            print(f"Warning: Samplerate mismatch ({sr} vs {samplerate}) for {path}. Attempting to append anyway.")
            
        # Append audio data
        combined_audio.append(data)
        
        # Add silence gap between segments (except after the last segment)
        if path != input_paths[-1] and silence_duration > 0:
            num_silence_samples = int(silence_duration * samplerate)
            if channels > 1:
                silence = np.zeros((num_silence_samples, channels))
            else:
                silence = np.zeros(num_silence_samples)
            combined_audio.append(silence)
            
    if not combined_audio:
        return False
        
    try:
        # Concatenate all arrays
        final_audio = np.concatenate(combined_audio, axis=0)
        
        # Normalize volume level (scale peak to 0.95 to avoid clipping)
        peak = np.max(np.abs(final_audio))
        if peak > 0:
            final_audio = (final_audio / peak) * 0.95
            
        # Ensure parent directory of output exists
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            
        # Write the final file
        sf.write(output_path, final_audio, samplerate)
        
        # Clean up input temporary files
        for path in input_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as delete_error:
                print(f"Warning: Could not remove temporary segment {path}: {delete_error}")
                
        return True
    except Exception as e:
        print(f"Error during audio composition: {e}")
        return False


def apply_audio_post_processing(
    audio_path: str,
    target_lufs: float = -14.0,
    fade_in_duration: float = 0.1,
    fade_out_duration: float = 0.1
) -> bool:
    """Applies LUFS loudness normalization and linear fade-in/fade-out transitions to a WAV file.
    
    If pyloudnorm is not installed, it falls back to standard peak normalization.
    """
    if not os.path.exists(audio_path):
        print(f"Error: WAV file not found for post-processing: {audio_path}")
        return False
        
    try:
        data, sr = sf.read(audio_path)
        
        # 1. Apply linear fade-in and fade-out
        num_samples = len(data)
        if num_samples > 0:
            # Fade-in
            if fade_in_duration > 0:
                fade_in_samples = int(fade_in_duration * sr)
                if fade_in_samples > num_samples:
                    fade_in_samples = num_samples
                # Generate linear fade curve (0.0 to 1.0)
                fade_in_curve = np.linspace(0.0, 1.0, fade_in_samples)
                # Reshape for broadcasting if stereo
                if len(data.shape) > 1:
                    fade_in_curve = fade_in_curve[:, np.newaxis]
                data[:fade_in_samples] *= fade_in_curve
                
            # Fade-out
            if fade_out_duration > 0:
                fade_out_samples = int(fade_out_duration * sr)
                if fade_out_samples > num_samples:
                    fade_out_samples = num_samples
                # Generate linear fade curve (1.0 to 0.0)
                fade_out_curve = np.linspace(1.0, 0.0, fade_out_samples)
                # Reshape for broadcasting if stereo
                if len(data.shape) > 1:
                    fade_out_curve = fade_out_curve[:, np.newaxis]
                data[-fade_out_samples:] *= fade_out_curve
                
        # 2. LUFS Normalization (requires pyloudnorm)
        if target_lufs is not None:
            try:
                import pyloudnorm as pyln
                # Create BS.1770 meter
                meter = pyln.Meter(sr)
                # Measure loudness
                loudness = meter.integrated_loudness(data)
                # Normalize only if we get a finite loudness reading
                if np.isfinite(loudness):
                    data = pyln.normalize.loudness(data, loudness, target_lufs)
                    # Limit peaks to 0.98 to prevent digital clipping distortion and pyloudnorm warnings
                    peak = np.max(np.abs(data))
                    if peak > 0.98:
                        data = (data / peak) * 0.98
                else:
                    # Fallback to standard peak normalization
                    peak = np.max(np.abs(data))
                    if peak > 0:
                        data = (data / peak) * 0.95
            except ImportError:
                # Fallback to standard peak normalization
                peak = np.max(np.abs(data))
                if peak > 0:
                    data = (data / peak) * 0.95
            except Exception as norm_err:
                print(f"Warning: LUFS normalization failed: {norm_err}. Falling back to peak normalization.")
                peak = np.max(np.abs(data))
                if peak > 0:
                    data = (data / peak) * 0.95
                
        # Save the processed audio back to the same file
        sf.write(audio_path, data, sr)
        return True
    except Exception as e:
        print(f"Error during audio post-processing: {e}")
        return False

