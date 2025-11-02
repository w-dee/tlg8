#!/usr/bin/env python3
import time, argparse
import torch

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=int, default=60, help="stress duration")
    p.add_argument("--n", type=int, default=8192, help="matrix size (n x n)")
    p.add_argument("--dtype", choices=["fp16","fp32"], default="fp16")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision("high")  # allow TF32 when using fp32
    use_amp = (args.dtype == "fp16")
    autocast_dtype = torch.float16

    n = args.n
    # 2つの大きな行列（勾配を流すので requires_grad=True）
    a = torch.randn(n, n, device=device, dtype=torch.float32, requires_grad=True)
    b = torch.randn(n, n, device=device, dtype=torch.float32, requires_grad=True)

    # メモ: 計算は fp16(autocast) か fp32 で行う
    end_t = time.time() + args.seconds
    it = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    while time.time() < end_t:
        a.grad = None
        b.grad = None
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_amp):
            # GEMM を複数回重ねて SM をしっかり使う
            c = a @ b          # [n, n]
            d = torch.tanh(c)  # 非線形
            e = d @ b          # もう一回 GEMM
            loss = (e * e).mean()
        loss.backward()
        it += 1

    torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1000
    print(f"done: iters={it}, avg_iter_ms={dt_ms/it:.1f} ms, n={n}, amp={use_amp}")

if __name__ == "__main__":
    main()

