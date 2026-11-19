# kernels

`matmul.py`: tiled Triton GEMM based on the official tutorial. FP16 in, FP32 accumulate. Every schedule knob is exposed as a launch parameter.
