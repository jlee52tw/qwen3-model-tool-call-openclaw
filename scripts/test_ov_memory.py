"""Test OpenVINO FP16 model inference at various sequence lengths to find memory ceiling."""
import openvino as ov
import numpy as np
import gc
import time
import os

MODEL_PATH = "C:/working/models/Qwen3-Coder-30B-A3B-Instruct/FP16-temp/openvino_model.xml"

def get_mem_gb():
    """Get current process RSS in GB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024**3)
    except ImportError:
        return -1

def get_system_free_gb():
    """Get system free RAM in GB."""
    try:
        import psutil
        return psutil.virtual_memory().available / (1024**3)
    except ImportError:
        return -1

def test_seq_len(core, model, seq_len):
    """Test inference at a given sequence length."""
    gc.collect()
    print(f"\n{'='*60}")
    print(f"Testing seq_len={seq_len}")
    print(f"  System free RAM: {get_system_free_gb():.1f} GB")
    print(f"  Process RSS before compile: {get_mem_gb():.1f} GB")
    
    compiled = core.compile_model(model, "CPU")
    infer_req = compiled.create_infer_request()
    print(f"  Process RSS after compile:  {get_mem_gb():.1f} GB")
    
    input_data = {
        "input_ids": np.ones((1, seq_len), dtype=np.int64),
        "attention_mask": np.ones((1, seq_len), dtype=np.int64),
        "position_ids": np.arange(seq_len, dtype=np.int64).reshape(1, -1),
        "beam_idx": np.zeros(1, dtype=np.int32),
    }
    
    t0 = time.time()
    try:
        result = infer_req.infer(input_data, share_inputs=True)
        elapsed = time.time() - t0
        logits = result[compiled.output(0)]
        print(f"  SUCCESS in {elapsed:.1f}s")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Process RSS after infer:   {get_mem_gb():.1f} GB")
        print(f"  System free RAM:           {get_system_free_gb():.1f} GB")
        del result, logits
        success = True
    except Exception as e:
        elapsed = time.time() - t0
        err = str(e)
        if "allocate" in err:
            import re
            m = re.search(r"allocate (\d+) bytes", err)
            if m:
                gb = int(m.group(1)) / (1024**3)
                print(f"  OOM after {elapsed:.1f}s: failed to allocate {gb:.1f} GB")
            else:
                print(f"  OOM after {elapsed:.1f}s: {err[:300]}")
        else:
            print(f"  FAILED after {elapsed:.1f}s: {err[:300]}")
        success = False
    
    del compiled, infer_req
    gc.collect()
    return success


if __name__ == "__main__":
    core = ov.Core()
    print("Reading model...")
    model = core.read_model(MODEL_PATH)
    print(f"Model loaded. Process RSS: {get_mem_gb():.1f} GB")
    
    # Test from small to large
    for seq_len in [128, 256, 512, 1024, 2048]:
        ok = test_seq_len(core, model, seq_len)
        if not ok:
            print(f"\n>>> Max safe seq_len for pure OV inference: < {seq_len}")
            # Try to find exact boundary
            lo, hi = seq_len // 2, seq_len
            while hi - lo > 64:
                mid = (lo + hi) // 2
                if test_seq_len(core, model, mid):
                    lo = mid
                else:
                    hi = mid
            print(f"\n>>> Approximate max safe seq_len: ~{lo}")
            break
    else:
        print("\n>>> All tests passed up to 2048 tokens!")
