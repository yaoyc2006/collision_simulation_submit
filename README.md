# Cloth / Collision Simulation

## File Overview

| File | Description |
|------|-------------|
| `main.py` | Entry point. Sets up cloth/collider scene and runs simulation loop. |
| `benchmark.py` | Performance benchmarks: CPU vs CUDA, spatial-hash vs brute-force. |
| `config.py` | Simulation parameters (time step, gravity, cloth size, etc.). |
| `objects.py` | CPU physics: `ParticleSystem`, `Cloth`, `MeshCloth`, `MeshCollider`, `SphereBody`. |
| `spatial_hash.py` | Spatial-hash structure for neighbor queries / collision acceleration. |
| `objects_cuda.py` | GPU physics: CUDA kernels, `ParticleSystemCUDA`, `MeshClothCUDA`, `MeshColliderCUDA`, and spatial-hash helpers. |
| `mesh_generator.py` | Generates simple meshes (plane, curtain) and exports to OBJ. |
| `render.py` | Blender import script for visualizing exported OBJ sequences. |
| `utils.py` | Helpers: OBJ export/load functions. |
| `benchmark_exp1_cuda_vs_cpu.csv` | Benchmark results: CUDA vs CPU timing. |
| `benchmark_exp2_hash_vs_brute.csv` | Benchmark results: spatial-hash vs brute-force. |
| `benchmark_results.png` | Benchmark visualization chart. |
| `collision_sim_final.mp4` | Video showing example simulation results and rendered output. |


## Requirements

- Python 3.8+
- numpy
- (Optional) numba + CUDA toolkit for GPU acceleration
 
## Configuration & Usage

This section groups the configuration and run-time usage notes for adjusting parameters, modes, mesh imports and initial conditions.

### Parameters

Where to change core simulation parameters and some common examples:

- `config.py`:
	- `dt` — time step per sub-step (e.g. `0.0001`).
	- `total_frames` / `sub_steps` — control how long and how finely each frame is simulated.
	- `gravity` — gravity vector (e.g. `np.array([0, -9.8, 0])`).
	- Cloth-specific: `k_stiffness`, `support_scale`. 
	- Collision: `self_collision_thickness`, `mesh_collision_iters`, `self_collision_iters`.

- `main.py`:
	- You can pass different `scale`, `offset`, and `rotation` when creating `MeshCloth` / `MeshCollider` objects.
	- Example: `cloth = MeshCloth("curtain_simple.obj", scale=6.0, offset=[0,11,10])`.

### Modes (CPU / CUDA / Collider type)

- Toggle CUDA acceleration in `main.py` by setting `use_cuda = True` or `False`.
- Toggle collider type by setting `use_mesh_collider = True` to use a mesh collider, otherwise the example uses a `SphereBody`.
- Implementation details live in `objects.py` (CPU) and `objects_cuda.py` (GPU). If `use_cuda=True` but CUDA isn't available, the code may raise errors — ensure `numba` and CUDA are installed.

### Importing / Using Mesh Files

- Place your OBJ files in the project root (same folder as `main.py`) or provide a relative path when instantiating `MeshCloth` / `MeshCollider`.
- Use `mesh_generator.py` to create simple meshes (plane/curtain). Example:

```bash
python mesh_generator.py
```

- Example usage in `main.py`:

```python
cloth = MeshCloth("curtain_simple.obj", scale=6.0, offset=[0,11,10], rotation=[0,0,0])
collider = MeshCollider("bunny_200.obj", scale=10.0, offset=[0,10,0], rotation=[0,0,0], thickness=0.15)
```

- If an OBJ is missing, `main.py` falls back to a default grid cloth (`Cloth()`).

### Initial conditions

- Initial velocities and mass for colliders and cloth can be set in `config.py` or directly in `main.py` when creating objects:
	- `collider_initial_velocity` and `collider_is_static` are in `config.py`.
	- `SphereBody(pos=[x,y,z], vel=[vx,vy,vz], radius=r, mass=m)` — directly set initial position/velocity/mass.
	- Cloth initial velocity can be set via `Config.cloth_initial_velocity`.

- Example: set a fast-moving collider in `config.py` or before the sim loop:

```python
# in config.py
collider_initial_velocity = np.array([0.0, 0.0, 5.0])

# or in main.py before the loop
sphere = SphereBody(pos=[0,8,0], vel=[0,0,10], radius=2, mass=4)
```

### Quick run checklist

- Edit `config.py` for global defaults.
- Edit `main.py` to choose run-time modes (`use_cuda`, `use_mesh_collider`) and per-instance transforms.
- Put OBJ meshes in the project folder and reference them by filename in `main.py`.
- Run:

```bash
python main.py
```

### Render / Visualization (using `render.py`)

- Purpose: `render.py` is a Blender import script that loads the simulation's exported OBJ sequence (cloth frames), `sphere_pos.txt` (sphere trajectory) and `collider_transform.txt` (collider transforms), sets up camera/lighting/materials and inserts keyframes so you can render the animation in Blender.

- Main configurable items (edit at the top of `render.py`):
	- `OBJ_DIR`: directory containing the OBJ sequence and exported text files (use absolute paths or paths relative to the script).
	- `SPHERE_FILE`: sphere position file (typically `sphere_pos.txt`).
	- `COLLIDER_FILE`: collider transform file (typically `collider_transform.txt`).
	- `SOLID_MESH_PATH` / `SOLID_MESH_SCALE`: path and scale for an optional solid mesh (e.g. the bunny) to import and display.
	- Camera/render settings: `CAMERA_DISTANCE`, `CAMERA_HEIGHT`, `CAMERA_LENS`, and render engine (script sets `CYCLES`).

- How to use:
	1. After the simulation finishes, confirm that the `.obj` frame sequence is present in `OBJ_DIR`, and that `sphere_pos.txt` or `collider_transform.txt` exist if you exported them.
	2. Edit the paths and parameters at the top of `render.py` as needed (absolute paths are recommended to avoid Blender working-directory issues).
	3. Run the script with Blender. Example command:

```bash
& "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" --python render.py
```

- Extra tips:
	- To animate a solid collider, enable `import_solid_mesh(scale=SOLID_MESH_SCALE)` in the script (it is commented out by default); the script reads `collider_transform.txt` to add keyframes.
	- The script attempts to use GPU (`bpy.context.scene.cycles.device = 'GPU'`); modify or remove that line to force CPU rendering if needed.
	- If Blender fails to find files, double-check `OBJ_DIR` and `SOLID_MESH_PATH` are correct and readable.
	- Confirm render scale (Important):
		- Set `SOLID_MESH_SCALE` 、 in `render.py` to match the `scale` parameter used in `main.py` when creating  `MeshCollider`.
		- Example: if `main.py` uses `MeshCollider(..., scale=20.0, ...)`, then use `SOLID_MESH_SCALE = 20.0` for the solid collider.