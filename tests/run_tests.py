import os
import sys
import subprocess
import shutil
import time
import json

def run_command(cmd_args: list[str]) -> tuple[int, str, str]:
    """Runs a command and returns exit code, stdout, and stderr."""
    # Run using the current virtualenv Python executable
    full_cmd = [sys.executable] + cmd_args
    try:
        result = subprocess.run(
            full_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            timeout=180 # 3 minutes timeout for heavy models like XTTSv2
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as e:
        return -1, "", f"Timeout expired: {e}"
    except Exception as e:
        return -1, "", str(e)

def format_duration(seconds: float) -> str:
    return f"{seconds:.2f}s"

def print_banner(text: str):
    print("\n" + "=" * 80)
    print(f" {text} ".center(80, "="))
    print("=" * 80)

def main():
    start_time = time.time()
    
    # Define directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tests_dir = os.path.join(base_dir, "tests")
    inputs_dir = os.path.join(tests_dir, "test_data", "inputs")
    outputs_dir = os.path.join(tests_dir, "test_data", "outputs")
    
    # Ensure directories exist
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Write temp config file for config override test
    temp_config_path = os.path.join(tests_dir, "test_data", "temp_config.json")
    temp_config = {
        "engine": "edge",
        "voice": "vi-VN-HoaiMyNeural",
        "speed": 1.15,
        "normalize": True,
        "fade_in": 0.2,
        "fade_out": 0.2,
        "silence_duration": 0.4
    }
    with open(temp_config_path, "w", encoding="utf-8") as f:
        json.dump(temp_config, f, indent=2)
        
    print_banner("AIVOICE TEST SUITE RUNNER")
    print(f"Base Directory: {base_dir}")
    print(f"Tests Directory: {tests_dir}")
    print(f"Python Executable: {sys.executable}")
    
    # Define test cases
    test_cases = [
        {
            "name": "test_edge_default",
            "desc": "Edge-TTS online engine with default Vietnamese voice",
            "args": ["main.py", "--input", "tests/test_data/inputs/test_vi.md", "--engine", "edge", "--voice", "vi-VN-NamMinhNeural"],
            "expected_output_name": "test_vi.wav",
            "expect_success": True
        },
        {
            "name": "test_edge_hoaimy_fast",
            "desc": "Edge-TTS with HoaiMy neural voice at 1.10x speed without normalization",
            "args": [
                "main.py", 
                "--input", "tests/test_data/inputs/test_vi.md", 
                "--engine", "edge", 
                "--voice", "vi-VN-HoaiMyNeural", 
                "--speed", "1.10",
                "--no-normalize"
            ],
            "expected_output_name": "test_vi.wav",
            "expect_success": True
        },
        {
            "name": "test_piper_default",
            "desc": "Piper engine with local vais1000 ONNX model",
            "args": [
                "main.py", 
                "--input", "tests/test_data/inputs/test_vi.md", 
                "--engine", "piper", 
                "--model", "models/piper/vi_VN-vais1000-medium.onnx"
            ],
            "expected_output_name": "test_vi.wav",
            "expect_success": True
        },
        {
            "name": "test_piper_slow",
            "desc": "Piper at 0.8x speed with custom fades (0.3s fade-in/out)",
            "args": [
                "main.py", 
                "--input", "tests/test_data/inputs/test_vi.md", 
                "--engine", "piper", 
                "--model", "models/piper/vi_VN-vais1000-medium.onnx",
                "--speed", "0.8",
                "--fade_in", "0.3",
                "--fade_out", "0.3"
            ],
            "expected_output_name": "test_vi.wav",
            "expect_success": True
        },
        {
            "name": "test_clone_vietnamese",
            "desc": "XTTSv2 local cloning using ref_voice.wav in Vietnamese (vi)",
            "args": [
                "main.py", 
                "--input", "tests/test_data/inputs/test_vi.md", 
                "--engine", "clone", 
                "--model", "models/xtts_v2", 
                "--ref_audio", "data/voices/ref_voice.wav", 
                "--voice", "vi"
            ],
            "expected_output_name": "test_vi.wav",
            "expect_success": True
        },
        {
            "name": "test_clone_english",
            "desc": "XTTSv2 local cloning using ref_voice.wav in English (en)",
            "args": [
                "main.py", 
                "--input", "tests/test_data/inputs/test_en.md", 
                "--engine", "clone", 
                "--model", "models/xtts_v2", 
                "--ref_audio", "data/voices/ref_voice.wav", 
                "--voice", "en"
            ],
            "expected_output_name": "test_en.wav",
            "expect_success": True
        },
        {
            "name": "test_phonemize",
            "desc": "XTTSv2 cloning with Vietnamese phonemizer enabled",
            "args": [
                "main.py", 
                "--input", "tests/test_data/inputs/test_vi.md", 
                "--engine", "clone", 
                "--model", "models/xtts_v2", 
                "--ref_audio", "data/voices/ref_voice.wav", 
                "--voice", "vi", 
                "--phonemize"
            ],
            "expected_output_name": "test_vi.wav",
            "expect_success": True
        },
        {
            "name": "test_batch_processing",
            "desc": "Batch processing mode on tests/test_data/inputs dir using Edge-TTS",
            "args": [
                "main.py", 
                "--input_dir", "tests/test_data/inputs", 
                "--engine", "edge", 
                "--voice", "vi-VN-NamMinhNeural",
                "--output_dir", "tests/test_data/outputs/batch_output"
            ],
            "expected_output_name": "batch_output", # verified separately
            "expect_success": True
        },
        {
            "name": "test_config_override",
            "desc": "Edge-TTS using custom config.json configuration override",
            "args": [
                "main.py", 
                "--input", "tests/test_data/inputs/test_vi.md", 
                "--config", "tests/test_data/temp_config.json"
            ],
            "expected_output_name": "test_vi.wav",
            "expect_success": True
        },
        {
            "name": "test_rvc_standalone",
            "desc": "RVC standalone voice conversion with local model weights",
            "args": [
                "main.py", 
                "--input", "data/voices/ref_voice.wav", 
                "--engine", "rvc", 
                "--rvc_model", "models/rvc/ElevenLabs_Adam_FR.pth", 
                "--rvc_index", "models/rvc/added_IVF4988_Flat_nprobe_1_ElevenLabs_Adam_FR_v2.index"
            ],
            "expected_output_name": "ref_voice.wav",
            "expect_success": True
        },
        {
            "name": "test_invalid_engine_error",
            "desc": "Negative test: passing an invalid engine name",
            "args": ["main.py", "--input", "tests/test_data/inputs/test_vi.md", "--engine", "invalid_engine_name"],
            "expected_output_name": None,
            "expect_success": False
        }
    ]
    
    results = []
    
    for idx, tc in enumerate(test_cases):
        name = tc["name"]
        desc = tc["desc"]
        args = tc["args"]
        expect_success = tc["expect_success"]
        
        print(f"\n[Test {idx+1}/{len(test_cases)}] {name} - {desc}")
        print(f"Command args: {' '.join(args)}")
        
        # Make sure no stale files in workspace root
        for f in ["test_vi.wav", "test_en.wav"]:
            if os.path.exists(os.path.join(base_dir, f)):
                try:
                    os.remove(os.path.join(base_dir, f))
                except OSError:
                    pass
                    
        t0 = time.time()
        exit_code, stdout, stderr = run_command(args)
        duration = time.time() - t0
        
        print(f"Duration: {format_duration(duration)} | Exit code: {exit_code}")
        
        # Evaluation
        passed = True
        error_msg = ""
        output_file_created = False
        output_size = 0
        
        if expect_success:
            if exit_code != 0:
                passed = False
                error_msg = f"Command failed with non-zero exit code: {exit_code}."
            else:
                # Check for output file
                if name == "test_batch_processing":
                    # Batch mode output verification
                    batch_dir = os.path.join(outputs_dir, "batch_output")
                    out1 = os.path.join(batch_dir, "test_vi", "test_vi.wav")
                    out2 = os.path.join(batch_dir, "test_en", "test_en.wav")
                    if os.path.exists(out1) and os.path.exists(out2):
                        output_file_created = True
                        output_size = os.path.getsize(out1) + os.path.getsize(out2)
                    else:
                        passed = False
                        error_msg = "Batch output directory/files not generated correctly."
                else:
                    # Single file output verification
                    gen_name = tc["expected_output_name"]
                    
                    # Calculate default folder structure: output/<input_base_name>/<gen_name>
                    input_val = None
                    for i in range(len(tc["args"]) - 1):
                        if tc["args"][i] == "--input":
                            input_val = tc["args"][i+1]
                            break
                    if input_val:
                        input_base_name = os.path.splitext(os.path.basename(input_val))[0]
                        gen_path = os.path.join(base_dir, "data", "outputs", input_base_name, gen_name)
                    else:
                        gen_path = os.path.join(base_dir, gen_name)
                    
                    if os.path.exists(gen_path):
                        output_file_created = True
                        output_size = os.path.getsize(gen_path)
                        
                        # Move to specific test output directory
                        case_out_dir = os.path.join(outputs_dir, name)
                        os.makedirs(case_out_dir, exist_ok=True)
                        dest_path = os.path.join(case_out_dir, f"{name}.wav")
                        
                        # Clean destination if it exists
                        if os.path.exists(dest_path):
                            os.remove(dest_path)
                            
                        shutil.move(gen_path, dest_path)
                        print(f"Moved output file to: {dest_path}")
                    else:
                        passed = False
                        error_msg = f"Expected output file '{gen_name}' was not created."
        else:
            # Negative test: expecting failure
            if exit_code == 0:
                passed = False
                error_msg = "Command succeeded, but was expected to fail."
                # Clean up if file got created by accident
                gen_name = tc.get("expected_output_name")
                if gen_name and os.path.exists(os.path.join(base_dir, gen_name)):
                    os.remove(os.path.join(base_dir, gen_name))
            else:
                print("Command failed as expected.")
                
        status = "PASSED" if passed else "FAILED"
        print(f"Result: {status}")
        if not passed:
            print(f"Error details: {error_msg}")
            if stderr:
                print(f"Stderr:\n{stderr.strip()}")
            if stdout:
                print(f"Stdout:\n{stdout.strip()[:500]}...") # truncate if long
                
        results.append({
            "name": name,
            "desc": desc,
            "status": status,
            "duration": duration,
            "output_size_kb": round(output_size / 1024.0, 1) if output_file_created else 0.0,
            "error_msg": error_msg
        })
        
    # Clean up temp config file
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)
        
    # Clean up any residual wav files in workspace root
    for f in ["test_vi.wav", "test_en.wav"]:
        if os.path.exists(os.path.join(base_dir, f)):
            try:
                os.remove(os.path.join(base_dir, f))
            except OSError:
                pass

    # Print summary table
    print_banner("SUMMARY REPORT")
    
    col_name = max(len("Test Name"), max(len(r["name"]) for r in results))
    col_status = len("Status")
    col_time = len("Time (s)")
    col_size = len("Size (KB)")
    col_desc = max(len("Description"), max(len(r["desc"]) for r in results))
    
    header = f"{'Test Name':<{col_name}}  {'Status':<{col_status}}  {'Time (s)':>{col_time}}  {'Size (KB)':>{col_size}}  {'Description':<{col_desc}}"
    print(header)
    print("-" * len(header))
    
    passed_count = 0
    for r in results:
        t_str = f"{r['duration']:.2f}"
        sz_str = f"{r['output_size_kb']:.1f}"
        print(f"{r['name']:<{col_name}}  {r['status']:<{col_status}}  {t_str:>{col_time}}  {sz_str:>{col_size}}  {r['desc']:<{col_desc}}")
        if r["status"] == "PASSED":
            passed_count += 1
            
    print("-" * len(header))
    print(f"Total: {len(results)} | Passed: {passed_count} | Failed: {len(results) - passed_count}")
    print(f"Total execution time: {format_duration(time.time() - start_time)}")
    print("=" * 80)
    
    # Exit with code 0 if all passed, otherwise 1
    if passed_count == len(results):
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
