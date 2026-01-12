# Cloth / Collision Simulation

## File Overview

| File | Description |
|------|-------------|
| `main.py` | Entry point. Sets up cloth/collider scene and runs simulation loop. |
| `benchmark.py` | Performance benchmarks: CPU vs CUDA, spatial-hash vs brute-force. |
| `config.py` | Simulation parameters (time step, gravity, cloth size, etc.). |
| `objects.py` | CPU physics: `ParticleSystem`, `Cloth`, `MeshCloth`, `MeshCollider`, `SphereBody`. |
| `objects_cuda.py` | GPU (CUDA) physics kernels and wrappers using Numba. |
| `spatial_hash.py` | Spatial-hash structure for neighbor queries / collision acceleration. |
| `mesh_generator.py` | Generates simple meshes (plane, curtain) and exports to OBJ. |
| `render.py` | Blender import script for visualizing exported OBJ sequences. |
| `utils.py` | Helpers: OBJ export/load functions. |
| `benchmark_exp1_cuda_vs_cpu.csv` | Benchmark results: CUDA vs CPU timing. |
| `benchmark_exp2_hash_vs_brute.csv` | Benchmark results: spatial-hash vs brute-force. |
| `benchmark_results.png` | Benchmark visualization chart. |

## Usage

```bash
python main.py        # Run simulation
python benchmark.py   # Run benchmarks
```

## Requirements

- Python 3.8+
- numpy
- (Optional) numba + CUDA toolkit for GPU acceleration
