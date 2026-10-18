Motion-context temporal anchor / payload patch logic in this build is adapted
from NikoDemon80/ComfyUI-H3-Motion-Context:
https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context

Upstream license: GPL-3.0.
The adaptation removes disk Save/Load and connects the previous sampled H3 latent
directly in RAM inside one ComfyUI DAG.

The optional Image Color Match feature reproduces the mean/std color transfer that
ComfyUI_essentials (https://github.com/cubiq/ComfyUI_essentials, MIT License,
Copyright (c) 2023 Matteo Spinelli) exposes through its ImageColorMatch node.
The implementation here is written from scratch for tensor batches and reuses the
kornia color-space conversions that ComfyUI already installs.
