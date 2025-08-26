# Experiments

These helper scripts reproduce common ablations and presets on macOS (M1 Max 32GB) or Linux.

Make scripts executable:
```bash
chmod +x experiments/*.sh
```

Fast preset (full mode, nearest-exact):
```bash
experiments/run_full_fast.sh /absolute/path/MMG.dcm
```

Quality preset (full mode, nearest-exact):
```bash
experiments/run_full_quality.sh /absolute/path/MMG.dcm
```

Sweep uncertainty samples (requires trained model path):
```bash
experiments/sweep_nsamples.sh /absolute/path/MMG.dcm Results/<run_dir>/model.pth
```

Zero-shot (requires pretrained prior):
```bash
experiments/run_zs.sh /absolute/path/MMG.dcm prior.pth
```

Notes:
- Consider `export PYTORCH_ENABLE_MPS_FALLBACK=1` on macOS for stability.
- Adjust `inf_patch_size`/`inf_overlap` to trade speed vs seams.
- Use `nearest-exact` to avoid linear-interpolation smoothing.