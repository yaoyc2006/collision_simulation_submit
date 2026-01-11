import numpy as np

class Config:
    dt = 0.0001
    gravity = np.array([0, 0, 0])
    friction = 0.9999
    cloth_initial_velocity = np.array([0.0, 0.0, 0.0])
    collider_initial_velocity = np.array([0.0, 0.0, 5.0])
    collider_is_static = False
    
    total_frames = 120
    sub_steps = 200
    
    cloth_res = (20, 20)
    cloth_size = (3.0, 3.0) 
    k_stiffness = 1000.0
    tension_scale = 1.0
    support_scale = 0.1
    stretch_limit_iters = 1
    
    self_collision_thickness = 0.1
    self_collision_iters = 1
    mesh_cloth_collision_multiplier = 2.0
    mesh_collision_iters = 1