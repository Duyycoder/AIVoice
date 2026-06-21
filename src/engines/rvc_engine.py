import os
import sys
import gc
import dataclasses
import torch

# Monkey-patch torch.load to default weights_only=False to support PyTorch 2.6+ on legacy checkpoints
original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

torch.load = patched_torch_load


# Monkey-patch dataclasses._get_field to bypass Python 3.11 mutable default checks in fairseq
original_get_field = dataclasses._get_field

def patched_get_field(cls, name, type, kw_only):
    try:
        return original_get_field(cls, name, type, kw_only)
    except ValueError as e:
        if "mutable default" in str(e):
            # Retrieve the default value from the class dict
            val = cls.__dict__.get(name, dataclasses.MISSING)
            default_val = val.default if isinstance(val, dataclasses.Field) else val
            
            if default_val is not dataclasses.MISSING and default_val is not None:
                cls_attr = default_val.__class__
                # Try patching the class __hash__ (works for custom classes like fairseq config classes)
                try:
                    cls_attr.__hash__ = lambda self: 0
                    return original_get_field(cls, name, type, kw_only)
                except TypeError:
                    # Built-in type (list, dict, set)
                    pass
            
            # Fallback for built-in list/dict/set by temporarily swapping it out
            original_val = getattr(cls, name, dataclasses.MISSING)
            try:
                # Set a dummy hashable value (like None) on the class to let original_get_field succeed
                setattr(cls, name, None)
                f = original_get_field(cls, name, type, kw_only)
                # Restore the original mutable default to the returned Field and class
                if isinstance(original_val, dataclasses.Field):
                    f.default = original_val.default
                    f.default_factory = original_val.default_factory
                else:
                    f.default = original_val
                setattr(cls, name, original_val)
                return f
            except Exception:
                # If swapping failed, restore original and raise the ValueError
                if original_val is not dataclasses.MISSING:
                    setattr(cls, name, original_val)
                raise
        raise

dataclasses._get_field = patched_get_field


def apply_rvc(input_wav_path: str, output_wav_path: str, model_path: str, index_path: str = None, pitch_shift: int = 0, device: str = None) -> bool:
    """
    Applies RVC (Retrieval-based Voice Conversion) to the input audio file
    to convert the voice and saves it to output_wav_path.
    
    Tries to utilize GPU (CUDA) if available and cleans up memory afterward.
    """
    if not model_path:
        raise ValueError("model_path must be specified for RVC.")
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"RVC model not found at: {model_path}")
        
    if index_path and not os.path.exists(index_path):
        print(f"Warning: RVC index file not found at: {index_path}. Proceeding without index.")
        index_path = None
        
    rvc_instance = None
    try:
        # Dynamic import of RVCInference class
        from rvc_python.infer import RVCInference
        
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            
        print(f"Running RVC voice conversion on device: {device}...")
        print(f"  Input: {input_wav_path}")
        print(f"  Output: {output_wav_path}")
        print(f"  Model: {model_path}")
        if index_path:
            print(f"  Index: {index_path}")
        print(f"  Pitch shift: {pitch_shift}")
        
        # Instantiate RVCInference
        rvc_instance = RVCInference(
            device=device,
            model_path=model_path,
            index_path=index_path or "",
            version="v2"
        )
        
        # Set parameters
        rvc_instance.set_params(f0up_key=pitch_shift, f0method="rmvpe")
        
        # Run inference
        rvc_instance.infer_file(input_wav_path, output_wav_path)
        
        print("RVC voice conversion completed successfully.")
        return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"RVC Voice Conversion failed: {e}", file=sys.stderr)
        return False
    finally:
        # Unload model and release instance
        if rvc_instance is not None:
            try:
                rvc_instance.unload_model()
            except Exception:
                pass
            del rvc_instance
            
        # Force garbage collection
        gc.collect()
        # Clean CUDA VRAM if PyTorch is available
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("CUDA VRAM successfully cleared after RVC.")
        except ImportError:
            pass


def apply_rvc_to_segments(
    input_wav_paths: list[str], 
    output_wav_paths: list[str], 
    model_path: str, 
    index_path: str = None, 
    pitch_shift: int = 0,
    device: str = None
) -> bool:
    """
    Applies RVC (Retrieval-based Voice Conversion) to a list of segment files 
    using a single loaded model instance to prevent VRAM OOM and reduce overhead.
    
    Tries to utilize GPU (CUDA) if available and cleans up memory afterward.
    """
    if not model_path:
        raise ValueError("model_path must be specified for RVC.")
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"RVC model not found at: {model_path}")
        
    if index_path and not os.path.exists(index_path):
        print(f"Warning: RVC index file not found at: {index_path}. Proceeding without index.")
        index_path = None
        
    rvc_instance = None
    try:
        # Dynamic import of RVCInference class
        from rvc_python.infer import RVCInference
        
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            
        print(f"Running RVC voice conversion on device: {device} for {len(input_wav_paths)} segments...")
        print(f"  Model: {model_path}")
        if index_path:
            print(f"  Index: {index_path}")
        print(f"  Pitch shift: {pitch_shift}")
        
        # Instantiate RVCInference once
        rvc_instance = RVCInference(
            device=device,
            model_path=model_path,
            index_path=index_path or "",
            version="v2"
        )
        
        # Set parameters
        rvc_instance.set_params(f0up_key=pitch_shift, f0method="rmvpe")
        
        # Run inference sequentially on each segment
        from tqdm import tqdm
        for inp_path, out_path in tqdm(zip(input_wav_paths, output_wav_paths), total=len(input_wav_paths), desc="RVC conversion"):
            if not os.path.exists(inp_path):
                continue
            rvc_instance.infer_file(inp_path, out_path)
            
        print("RVC segment voice conversion completed successfully.")
        return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"RVC Voice Conversion failed: {e}", file=sys.stderr)
        return False
    finally:
        # Unload model and release instance
        if rvc_instance is not None:
            try:
                rvc_instance.unload_model()
            except Exception:
                pass
            del rvc_instance
            
        # Force garbage collection
        gc.collect()
        # Clean CUDA VRAM if PyTorch is available
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("CUDA VRAM successfully cleared after RVC.")
        except ImportError:
            pass
