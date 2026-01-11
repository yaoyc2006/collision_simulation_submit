import numpy as np
import os
from config import Config
from objects import Cloth, SphereBody, MeshCloth, MeshCollider
from objects_cuda import MeshClothCUDA, MeshColliderCUDA
from utils import export_frame

def solve_collision_coupling(cloth, sphere):
    diffs = cloth.particles.pos - sphere.pos
    dists = np.linalg.norm(diffs, axis=1)
    
    threshold = sphere.radius + 0.06
    collision_mask = dists < threshold
    
    if not np.any(collision_mask):
        return

    safe_dists = dists[collision_mask]
    safe_dists[safe_dists == 0] = 0.0001 
    normals = diffs[collision_mask] / safe_dists[:, np.newaxis]
    cloth.particles.pos[collision_mask] = sphere.pos + normals * threshold

    vels = cloth.particles.vel[collision_mask]
    sphere_vel_proj = np.sum(sphere.vel * normals, axis=1)[:, np.newaxis] * normals
    cloth.particles.vel[collision_mask] = sphere_vel_proj + (vels - sphere_vel_proj) * 0.9


    num_colliding = np.sum(collision_mask)
    drag_force = -np.sum(normals, axis=0) * num_colliding * 0.1
    
    sphere.vel += drag_force * sphere.inv_mass * Config.dt

'''
def main():
    if not os.path.exists("output"):
        os.makedirs("output")

    cloth = Cloth()
    sphere = SphereBody(pos=[0, 2, 0], vel=[0, 10, 0], radius=0.8, mass=500.0) 
    sphere_positions = [] 

    print(f"Start Simulation for {Config.total_frames} frames...")
    
    for frame in range(Config.total_frames):
        for _ in range(Config.sub_steps):
            cloth.update(Config.dt)
            sphere.update(Config.dt)
            solve_collision_coupling(cloth, sphere)
            
        export_frame(cloth, sphere, frame)
        
        sphere_positions.append(sphere.pos.copy()) 

    np.savetxt("output/sphere_pos.txt", sphere_positions, fmt="%.6f")
    print("Sphere positions saved to 'output/sphere_pos.txt'")
'''

def main():
    if not os.path.exists("output"):
        os.makedirs("output")

    use_mesh_collider = False
    use_cuda = False
    
    try:
        if use_cuda:
            cloth = MeshClothCUDA("curtain_simple.obj", scale=6.0, offset=[0, 11, 10],rotation=[0, 0, 0])
            print("Using CUDA-accelerated MeshClothCUDA")
        else:
            cloth = MeshCloth("curtain_simple.obj", scale=4.0, offset=[0, 5, 15], rotation=[0, 0, 0])
            print("Using CPU MeshCloth")
    except FileNotFoundError:
        print("Warning: OBJ not found, using default Grid Cloth.")
        cloth = Cloth()
        use_cuda = False
    
    collider = None
    collider_positions = []
    collider_rotations = []
    if use_mesh_collider:
        try:
            collider_offset = [0, 10, 0]
            collider_mass = 0.0 if Config.collider_is_static else 10
            #collider = MeshCollider("bunny_200.obj", scale=10.0, offset=collider_offset, rotation=[0, 0, 0], thickness=0.15, mass=collider_mass)
            collider = MeshColliderCUDA("bunny_200_subdivided_1.obj", scale=20.0, offset=collider_offset, rotation=[0, 0, 0], thickness=Config.self_collision_thickness, mass=collider_mass)
            collider.pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            collider.rot = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            collider.vel = np.array(Config.collider_initial_velocity, dtype=np.float64)
            collider.angular_vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            collider_type = "STATIC" if Config.collider_is_static else "DYNAMIC"
            print(f"Using {collider_type} mesh collider (mass={collider.mass} kg)")
        except FileNotFoundError:
            print("Warning: obstacle.obj not found, falling back to sphere collider.")
            collider = None

    sphere = SphereBody(pos=[0, 8, 0], vel=[0, 0, 0], radius=2, mass=4) 
    sphere_positions = [] 

    print(f"Start Simulation for {Config.total_frames} frames...")
    
    for frame in range(Config.total_frames):
        for _ in range(Config.sub_steps):
            cloth.update(Config.dt)
            sphere.update(Config.dt)
            
            if collider is not None:
                collider.collide(cloth.particles, iterations=getattr(Config, 'mesh_collision_iters', 2))
                if Config.collider_is_static:
                    collider.pos += collider.vel * Config.dt
                    collider.rot += collider.angular_vel * Config.dt
                else:
                    collider.update(Config.dt, gravity=Config.gravity)
            elif use_cuda and hasattr(cloth, 'solve_sphere_collision'):
                cloth.solve_sphere_collision(sphere)
            else:
                solve_collision_coupling(cloth, sphere)
            
        export_frame(cloth, sphere, frame, collider=collider)
        
        if collider is not None:
            collider_positions.append((collider.pos + collider.offset_init).copy())
            collider_rotations.append(collider.rot.copy())
        else:
            sphere_positions.append(sphere.pos.copy())

    if collider is not None:
        collider_data = np.column_stack([collider_positions, collider_rotations])
        np.savetxt("output/collider_transform.txt", collider_data, fmt="%.6f", 
                   header="x y z rx ry rz (position and rotation in degrees)", comments="")
        print("Collider transform saved to 'output/collider_transform.txt'")
    else:
        np.savetxt("output/sphere_pos.txt", sphere_positions, fmt="%.6f")
        print("Sphere positions saved to 'output/sphere_pos.txt'")
    
if __name__ == "__main__":
    main()