import numpy as np
from numba import cuda
import math
from config import Config
from utils import load_obj_mesh

@cuda.jit
def apply_gravity_kernel(forces, inv_mass, gravity, num_particles):
    i = cuda.grid(1)
    if i < num_particles:
        if inv_mass[i] > 0:
            mass = 1.0 / inv_mass[i]
            forces[i, 0] += mass * gravity[0]
            forces[i, 1] += mass * gravity[1]
            forces[i, 2] += mass * gravity[2]


@cuda.jit
def integrate_kernel(pos, vel, forces, inv_mass, dt, friction, num_particles):
    i = cuda.grid(1)
    if i < num_particles:
        fx = forces[i, 0]
        fy = forces[i, 1]
        fz = forces[i, 2]
        
        if math.isnan(fx) or math.isinf(fx):
            fx = 0.0
        if math.isnan(fy) or math.isinf(fy):
            fy = 0.0
        if math.isnan(fz) or math.isinf(fz):
            fz = 0.0
        
        acc_x = fx * inv_mass[i]
        acc_y = fy * inv_mass[i]
        acc_z = fz * inv_mass[i]
        
        vel[i, 0] = (vel[i, 0] + acc_x * dt) * friction
        vel[i, 1] = (vel[i, 1] + acc_y * dt) * friction
        vel[i, 2] = (vel[i, 2] + acc_z * dt) * friction
        
        if math.isnan(vel[i, 0]) or math.isinf(vel[i, 0]):
            vel[i, 0] = 0.0
        if math.isnan(vel[i, 1]) or math.isinf(vel[i, 1]):
            vel[i, 1] = 0.0
        if math.isnan(vel[i, 2]) or math.isinf(vel[i, 2]):
            vel[i, 2] = 0.0
        
        max_vel = 30.0
        speed_sq = vel[i, 0]**2 + vel[i, 1]**2 + vel[i, 2]**2
        if speed_sq > max_vel * max_vel:
            scale = max_vel / math.sqrt(speed_sq + 1e-8)
            vel[i, 0] *= scale
            vel[i, 1] *= scale
            vel[i, 2] *= scale
        
        pos[i, 0] += vel[i, 0] * dt
        pos[i, 1] += vel[i, 1] * dt
        pos[i, 2] += vel[i, 2] * dt
        
        forces[i, 0] = 0.0
        forces[i, 1] = 0.0
        forces[i, 2] = 0.0


@cuda.jit
def compute_spring_forces_kernel(pos, vel, forces, inv_mass,
                                  spring_indices, spring_data,
                                  damping_factor, tension_scale, support_scale,
                                  num_springs):
    s = cuda.grid(1)
    if s < num_springs:
        p1_idx = spring_indices[s, 0]
        p2_idx = spring_indices[s, 1]
        rest_len = spring_data[s, 0]
        k = spring_data[s, 1]
        
        dx = pos[p1_idx, 0] - pos[p2_idx, 0]
        dy = pos[p1_idx, 1] - pos[p2_idx, 1]
        dz = pos[p1_idx, 2] - pos[p2_idx, 2]
        
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        if dist < 1e-6:
            return
        
        dir_x = dx / dist
        dir_y = dy / dist
        dir_z = dz / dist
        
        raw_force = k * (dist - rest_len)
        
        if raw_force >= 0:
            raw_force *= tension_scale
        else:
            raw_force *= support_scale
        
        rel_vx = vel[p1_idx, 0] - vel[p2_idx, 0]
        rel_vy = vel[p1_idx, 1] - vel[p2_idx, 1]
        rel_vz = vel[p1_idx, 2] - vel[p2_idx, 2]
        
        damping = (rel_vx * dir_x + rel_vy * dir_y + rel_vz * dir_z) * damping_factor
        
        total_force = raw_force + damping
        
        fx = -dir_x * total_force
        fy = -dir_y * total_force
        fz = -dir_z * total_force
        
        cuda.atomic.add(forces, (p1_idx, 0), fx)
        cuda.atomic.add(forces, (p1_idx, 1), fy)
        cuda.atomic.add(forces, (p1_idx, 2), fz)
        cuda.atomic.add(forces, (p2_idx, 0), -fx)
        cuda.atomic.add(forces, (p2_idx, 1), -fy)
        cuda.atomic.add(forces, (p2_idx, 2), -fz)


@cuda.jit
def stretch_limit_kernel(pos, inv_mass, spring_indices, spring_data,
                         stretch_limit, num_springs):
    s = cuda.grid(1)
    if s < num_springs:
        p1_idx = spring_indices[s, 0]
        p2_idx = spring_indices[s, 1]
        rest_len = spring_data[s, 0]
        max_len = rest_len * stretch_limit
        
        dx = pos[p1_idx, 0] - pos[p2_idx, 0]
        dy = pos[p1_idx, 1] - pos[p2_idx, 1]
        dz = pos[p1_idx, 2] - pos[p2_idx, 2]
        
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        if dist > max_len and dist > 1e-6:
            correction = (dist - max_len) / dist * 0.5
            
            w1 = inv_mass[p1_idx]
            w2 = inv_mass[p2_idx]
            total_w = w1 + w2
            
            if total_w > 0:
                scale1 = w1 / total_w
                scale2 = w2 / total_w
                
                corr_x = dx * correction
                corr_y = dy * correction
                corr_z = dz * correction
                
                if w1 > 0:
                    cuda.atomic.add(pos, (p1_idx, 0), -corr_x * scale1)
                    cuda.atomic.add(pos, (p1_idx, 1), -corr_y * scale1)
                    cuda.atomic.add(pos, (p1_idx, 2), -corr_z * scale1)
                if w2 > 0:
                    cuda.atomic.add(pos, (p2_idx, 0), corr_x * scale2)
                    cuda.atomic.add(pos, (p2_idx, 1), corr_y * scale2)
                    cuda.atomic.add(pos, (p2_idx, 2), corr_z * scale2)


@cuda.jit
def ground_collision_kernel(pos, vel, ground_y, num_particles):
    i = cuda.grid(1)
    if i < num_particles:
        if pos[i, 1] < ground_y:
            pos[i, 1] = ground_y
            vel[i, 1] *= -0.1
            vel[i, 0] *= 0.95
            vel[i, 2] *= 0.95


@cuda.jit
def sphere_collision_kernel(pos, vel, inv_mass, 
                            sphere_x, sphere_y, sphere_z,
                            sphere_vx, sphere_vy, sphere_vz,
                            threshold, num_particles):
    i = cuda.grid(1)
    if i < num_particles:
        dx = pos[i, 0] - sphere_x
        dy = pos[i, 1] - sphere_y
        dz = pos[i, 2] - sphere_z
        
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        if dist < threshold and dist > 1e-6:
            nx = dx / dist
            ny = dy / dist
            nz = dz / dist
            
            pos[i, 0] = sphere_x + nx * threshold
            pos[i, 1] = sphere_y + ny * threshold
            pos[i, 2] = sphere_z + nz * threshold
            
            vx = vel[i, 0]
            vy = vel[i, 1]
            vz = vel[i, 2]
            
            vel[i, 0] = sphere_vx + (vx - sphere_vx) * 0.98
            vel[i, 1] = sphere_vy + (vy - sphere_vy) * 0.98
            vel[i, 2] = sphere_vz + (vz - sphere_vz) * 0.98

@cuda.jit(device=True)
def hash_cell(x, y, z, cell_size, grid_size):
    cx = int(math.floor(x / cell_size)) % grid_size
    cy = int(math.floor(y / cell_size)) % grid_size
    cz = int(math.floor(z / cell_size)) % grid_size
    if cx < 0: cx += grid_size
    if cy < 0: cy += grid_size
    if cz < 0: cz += grid_size
    return cx + cy * grid_size + cz * grid_size * grid_size


@cuda.jit
def compute_cell_indices_kernel(pos, cell_indices, cell_size, grid_size, num_particles):
    i = cuda.grid(1)
    if i < num_particles:
        cell_indices[i] = hash_cell(pos[i, 0], pos[i, 1], pos[i, 2], cell_size, grid_size)


@cuda.jit
def count_particles_per_cell_kernel(cell_indices, cell_counts, num_particles):
    i = cuda.grid(1)
    if i < num_particles:
        cell = cell_indices[i]
        cuda.atomic.add(cell_counts, cell, 1)


@cuda.jit
def clear_array_kernel(arr, size):
    i = cuda.grid(1)
    if i < size:
        arr[i] = 0


@cuda.jit
def compute_cell_offsets_kernel(cell_counts, cell_offsets, num_cells):
    # 这个kernel在GPU单线程运行非常慢, 用CPU版本替代
    i = cuda.grid(1)
    if i == 0:
        offset = 0
        for c in range(num_cells):
            cell_offsets[c] = offset
            offset += cell_counts[c]

_cell_counts_buffer = None
_cell_offsets_buffer = None

def compute_cell_offsets_cpu(cell_counts_device, cell_offsets_device, num_cells):
    global _cell_counts_buffer, _cell_offsets_buffer
    
    if _cell_counts_buffer is None or len(_cell_counts_buffer) < num_cells:
        _cell_counts_buffer = np.zeros(num_cells, dtype=np.int32)
        _cell_offsets_buffer = np.zeros(num_cells, dtype=np.int32)
    
    cell_counts_device.copy_to_host(_cell_counts_buffer[:num_cells])
    
    _cell_offsets_buffer[0] = 0
    _cell_offsets_buffer[1:num_cells] = np.cumsum(_cell_counts_buffer[:num_cells-1])

    cell_offsets_device.copy_to_device(_cell_offsets_buffer[:num_cells])


@cuda.jit
def sort_particles_kernel(cell_indices, cell_offsets, cell_counts_temp, 
                          sorted_indices, num_particles):
    i = cuda.grid(1)
    if i < num_particles:
        cell = cell_indices[i]
        local_idx = cuda.atomic.add(cell_counts_temp, cell, 1)
        sorted_indices[cell_offsets[cell] + local_idx] = i


@cuda.jit
def self_collision_spatial_hash_kernel(pos, vel, inv_mass, adjacency_matrix,
                                       cell_indices, sorted_indices, 
                                       cell_offsets, cell_counts,
                                       collision_thickness, cell_size, grid_size,
                                       num_particles, num_cells):
    i = cuda.grid(1)
    if i < num_particles:
        if inv_mass[i] <= 0:
            return
        
        px_i = pos[i, 0]
        py_i = pos[i, 1]
        pz_i = pos[i, 2]
        
        cx = int(math.floor(px_i / cell_size))
        cy = int(math.floor(py_i / cell_size))
        cz = int(math.floor(pz_i / cell_size))
        
        corr_x = 0.0
        corr_y = 0.0
        corr_z = 0.0
        collision_count = 0
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    ncx = (cx + dx) % grid_size
                    ncy = (cy + dy) % grid_size
                    ncz = (cz + dz) % grid_size
                    if ncx < 0: ncx += grid_size
                    if ncy < 0: ncy += grid_size
                    if ncz < 0: ncz += grid_size
                    
                    cell_idx = ncx + ncy * grid_size + ncz * grid_size * grid_size
                    
                    if cell_idx >= num_cells:
                        continue
                    
                    start = cell_offsets[cell_idx]
                    count = cell_counts[cell_idx]
                    
                    for k in range(count):
                        j = sorted_indices[start + k]
                        
                        if i == j:
                            continue
                        if inv_mass[j] <= 0:
                            continue

                        if adjacency_matrix[i, j] > 0:
                            continue

                        djx = px_i - pos[j, 0]
                        djy = py_i - pos[j, 1]
                        djz = pz_i - pos[j, 2]
                        
                        dist_sq = djx*djx + djy*djy + djz*djz

                        min_dist_sq = collision_thickness * collision_thickness * 4.0
                        if dist_sq < min_dist_sq and dist_sq > 1e-10:
                            dist = math.sqrt(dist_sq)
                            
                            nx = djx / dist
                            ny = djy / dist
                            nz = djz / dist
                            
                            min_dist = collision_thickness * 2.0
                            penetration = min_dist - dist
                            
                            w_i = inv_mass[i]
                            w_j = inv_mass[j]
                            total_w = w_i + w_j
                            
                            if total_w > 0:
                                ratio_i = w_i / total_w

                            corr_x += nx * penetration * ratio_i * 1.5
                            corr_y += ny * penetration * ratio_i * 1.5
                            corr_z += nz * penetration * ratio_i * 1.5
                            collision_count += 1
                            
                            rel_vx = vel[i, 0] - vel[j, 0]
                            rel_vy = vel[i, 1] - vel[j, 1]
                            rel_vz = vel[i, 2] - vel[j, 2]
                            v_normal = rel_vx * nx + rel_vy * ny + rel_vz * nz
                            
                            if v_normal < 0: 
                                damping = 0.8
                                vel[i, 0] -= nx * v_normal * damping * ratio_i
                                vel[i, 1] -= ny * v_normal * damping * ratio_i
                                vel[i, 2] -= nz * v_normal * damping * ratio_i
        
        if collision_count > 0:
            pos[i, 0] += corr_x
            pos[i, 1] += corr_y
            pos[i, 2] += corr_z

@cuda.jit
def self_collision_kernel(pos, vel, inv_mass, adjacency_matrix, 
                          collision_thickness, num_particles):  
    # 暴力版本，用于小网格
    i = cuda.grid(1)
    if i < num_particles:
        if inv_mass[i] <= 0:
            return
            
        px_i = pos[i, 0]
        py_i = pos[i, 1]
        pz_i = pos[i, 2]

        corr_x = 0.0
        corr_y = 0.0
        corr_z = 0.0
        collision_count = 0

        for j in range(num_particles):
            if i == j:
                continue
            if inv_mass[j] <= 0:
                continue
            
            if adjacency_matrix[i, j] > 0:
                continue
            
            dx = px_i - pos[j, 0]
            dy = py_i - pos[j, 1]
            dz = pz_i - pos[j, 2]
            
            dist_sq = dx*dx + dy*dy + dz*dz

            min_dist_sq = collision_thickness * collision_thickness * 4.0
            if dist_sq < min_dist_sq and dist_sq > 1e-10:
                dist = math.sqrt(dist_sq)

                nx = dx / dist
                ny = dy / dist
                nz = dz / dist

                min_dist = collision_thickness * 2.0
                penetration = min_dist - dist

                w_i = inv_mass[i]
                w_j = inv_mass[j]
                total_w = w_i + w_j
                
                if total_w > 0:
                    ratio_i = w_i / total_w
                    corr_x += nx * penetration * ratio_i * 1.5
                    corr_y += ny * penetration * ratio_i * 1.5
                    corr_z += nz * penetration * ratio_i * 1.5
                    collision_count += 1

                    rel_vx = vel[i, 0] - vel[j, 0]
                    rel_vy = vel[i, 1] - vel[j, 1]
                    rel_vz = vel[i, 2] - vel[j, 2]
                    v_normal = rel_vx * nx + rel_vy * ny + rel_vz * nz
                    
                    if v_normal < 0:
                        damping = 0.8
                        vel[i, 0] -= nx * v_normal * damping * ratio_i
                        vel[i, 1] -= ny * v_normal * damping * ratio_i
                        vel[i, 2] -= nz * v_normal * damping * ratio_i
        
        if collision_count > 0:
            pos[i, 0] += corr_x
            pos[i, 1] += corr_y
            pos[i, 2] += corr_z

class ParticleSystemCUDA:
    def __init__(self, num_particles):
        self.num = num_particles
        
        self.pos = cuda.to_device(np.zeros((num_particles, 3), dtype=np.float32))
        self.vel = cuda.to_device(np.zeros((num_particles, 3), dtype=np.float32))
        self.inv_mass = cuda.to_device(np.ones(num_particles, dtype=np.float32))
        self.forces = cuda.to_device(np.zeros((num_particles, 3), dtype=np.float32))

        self.threads_per_block = 256

        
        self.blocks = (num_particles + self.threads_per_block - 1) // self.threads_per_block

    def set_positions(self, pos_cpu):
        self.pos = cuda.to_device(pos_cpu.astype(np.float32))
    
    def set_inv_mass(self, inv_mass_cpu):
        self.inv_mass = cuda.to_device(inv_mass_cpu.astype(np.float32))

    def apply_gravity(self, gravity):
        gravity_gpu = cuda.to_device(np.array(gravity, dtype=np.float32))
        apply_gravity_kernel[self.blocks, self.threads_per_block](
            self.forces, self.inv_mass, gravity_gpu, self.num
        )

    def integrate(self, dt, friction):
        integrate_kernel[self.blocks, self.threads_per_block](
            self.pos, self.vel, self.forces, self.inv_mass,
            dt, friction, self.num
        )

    def to_numpy(self):
        return self.pos.copy_to_host()
    
    def get_velocities(self):
        return self.vel.copy_to_host()
    
    def set_positions_partial(self, indices, new_pos):
        pos_cpu = self.pos.copy_to_host()
        pos_cpu[indices] = new_pos.astype(np.float32)
        self.pos = cuda.to_device(pos_cpu)
    
    def set_velocities_partial(self, indices, new_vel):
        vel_cpu = self.vel.copy_to_host()
        vel_cpu[indices] = new_vel.astype(np.float32)
        self.vel = cuda.to_device(vel_cpu)


class MeshClothCUDA:
    def __init__(self, obj_path, scale=1.0, offset=(0, 0, 0), rotation=(0, 0, 0)):
        print(f"Loading mesh from {obj_path} (CUDA mode)...")
        raw_verts, raw_faces = load_obj_mesh(obj_path)
        
        self.faces = np.array(raw_faces, dtype=np.int32)
        num_particles = len(raw_verts)
        print(f"Mesh loaded: {num_particles} vertices, {len(raw_faces)} faces.")
        
        self.particles = ParticleSystemCUDA(num_particles)

        pos_cpu = raw_verts.astype(np.float32) * scale
        
        rx, ry, rz = np.deg2rad(rotation)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(rx), -np.sin(rx)],
                       [0, np.sin(rx), np.cos(rx)]], dtype=np.float32)
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                       [0, 1, 0],
                       [-np.sin(ry), 0, np.cos(ry)]], dtype=np.float32)
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                       [np.sin(rz), np.cos(rz), 0],
                       [0, 0, 1]], dtype=np.float32)
        R = Rz @ Ry @ Rx
        pos_cpu = pos_cpu @ R.T
        pos_cpu += np.array(offset, dtype=np.float32)
        
        self.particles.set_positions(pos_cpu)
        try:
            init_vel = np.asarray(Config.cloth_initial_velocity, dtype=np.float32)
            vel_cpu = np.tile(init_vel[np.newaxis, :], (num_particles, 1)).astype(np.float32)
            indices = np.arange(num_particles, dtype=np.int32)
            self.particles.set_velocities_partial(indices, vel_cpu)
        except Exception:
            pass
        
        total_mass = 2.0
        particle_mass = total_mass / num_particles
        inv_mass_cpu = np.ones(num_particles, dtype=np.float32) / particle_mass
        self.particles.set_inv_mass(inv_mass_cpu)

        spring_indices = []
        spring_data = []
        existing_links = set()
        
        def add_link(i, j, stiffness_ratio):
            if i == j: return
            if i > j: i, j = j, i
            if (i, j) in existing_links: return
            
            p1 = pos_cpu[i]
            p2 = pos_cpu[j]
            rest_len = np.linalg.norm(p1 - p2)
            if rest_len < 0.0001: return
            
            k = Config.k_stiffness * stiffness_ratio
            spring_indices.append([i, j])
            spring_data.append([rest_len, k])
            existing_links.add((i, j))

        print("Building Structural Springs...")
        edge_to_faces = {}
        for fi, face in enumerate(self.faces):
            n = len(face)
            for i in range(n):
                idx1 = face[i]
                idx2 = face[(i+1) % n]
                add_link(idx1, idx2, 1.0)
                edge_key = (min(idx1, idx2), max(idx1, idx2))
                if edge_key not in edge_to_faces:
                    edge_to_faces[edge_key] = []
                edge_to_faces[edge_key].append((fi, idx1, idx2))

        print("Building Bending Springs...")
        for edge_key, face_list in edge_to_faces.items():
            if len(face_list) == 2:
                fi1, e1_a, e1_b = face_list[0]
                fi2, e2_a, e2_b = face_list[1]
                face1 = self.faces[fi1]
                face2 = self.faces[fi2]
                opp1 = [v for v in face1 if v != e1_a and v != e1_b]
                opp2 = [v for v in face2 if v != e2_a and v != e2_b]
                if opp1 and opp2:
                    add_link(opp1[0], opp2[0], 0.3)
        
        self.num_springs = len(spring_indices)
        print(f"Springs built: {self.num_springs} constraints (CUDA).")

        self.spring_indices = cuda.to_device(np.array(spring_indices, dtype=np.int32))
        self.spring_data = cuda.to_device(np.array(spring_data, dtype=np.float32))

        config_thickness = getattr(Config, 'self_collision_thickness', 0.05)
        self.collision_thickness = config_thickness

        if self.num_springs > 0:
            edge_lens = [s[0] for s in spring_data]
            avg_edge_len = np.mean(edge_lens)
            min_edge_len = np.min(edge_lens)
            print(f"Mesh statistics: Avg edge length = {avg_edge_len:.4f}, Min edge length = {min_edge_len:.4f}")

            max_allowed_thickness = min_edge_len * 0.45
            
            if config_thickness > max_allowed_thickness:
                print(f"Adjusting self_collision_thickness from {config_thickness} to {max_allowed_thickness:.4f} based on mesh density.")
                self.collision_thickness = max_allowed_thickness

        self.enable_self_collision = True

        self._build_adjacency(spring_indices, num_particles)

        self.threads_per_block = 256
        self.spring_blocks = (self.num_springs + self.threads_per_block - 1) // self.threads_per_block
    
    def _build_adjacency(self, spring_indices, num_particles):
        adjacency = np.zeros((num_particles, num_particles), dtype=np.int8)
        
        for i, j in spring_indices:
            adjacency[i, j] = 1
            adjacency[j, i] = 1
               
        self.adjacency_gpu = cuda.to_device(adjacency)

        spring_arr = np.array(spring_indices, dtype=np.int32)
        self.spring_p1_gpu = cuda.to_device(spring_arr[:, 0].copy())
        self.spring_p2_gpu = cuda.to_device(spring_arr[:, 1].copy())
        
        self.use_spatial_hash = num_particles > 500  # 粒子多时使用空间哈希
        if self.use_spatial_hash:
            self._init_spatial_hash(num_particles)
    
    def _init_spatial_hash(self, num_particles):
        self.cell_size = self.collision_thickness * 1.5
        self.grid_size = 64
        self.num_cells = self.grid_size ** 3

        self.cell_indices = cuda.to_device(np.zeros(num_particles, dtype=np.int32))
        self.sorted_indices = cuda.to_device(np.zeros(num_particles, dtype=np.int32))
        self.cell_counts = cuda.to_device(np.zeros(self.num_cells, dtype=np.int32))
        self.cell_offsets = cuda.to_device(np.zeros(self.num_cells, dtype=np.int32))
        self.cell_counts_temp = cuda.to_device(np.zeros(self.num_cells, dtype=np.int32))
        
        print(f"Spatial hash initialized: grid {self.grid_size}^3, cell_size={self.cell_size:.3f}")
    
    def _update_spatial_hash(self):
        num_particles = self.particles.num
        blocks = self.particles.blocks
        tpb = self.threads_per_block

        cell_blocks = (self.num_cells + tpb - 1) // tpb
        clear_array_kernel[cell_blocks, tpb](self.cell_counts, self.num_cells)
        clear_array_kernel[cell_blocks, tpb](self.cell_counts_temp, self.num_cells)

        compute_cell_indices_kernel[blocks, tpb](
            self.particles.pos, self.cell_indices, 
            self.cell_size, self.grid_size, num_particles
        )

        count_particles_per_cell_kernel[blocks, tpb](
            self.cell_indices, self.cell_counts, num_particles
        )

        compute_cell_offsets_cpu(self.cell_counts, self.cell_offsets, self.num_cells)

        sort_particles_kernel[blocks, tpb](
            self.cell_indices, self.cell_offsets, self.cell_counts_temp,
            self.sorted_indices, num_particles
        )

    def solve_constraints(self):

        compute_spring_forces_kernel[self.spring_blocks, self.threads_per_block](
            self.particles.pos, self.particles.vel, self.particles.forces,
            self.particles.inv_mass, self.spring_indices, self.spring_data,
            3.0,  # damping_factor
            Config.tension_scale, Config.support_scale,
            self.num_springs
        )

        for _ in range(Config.stretch_limit_iters):
            stretch_limit_kernel[self.spring_blocks, self.threads_per_block](
                self.particles.pos, self.particles.inv_mass,
                self.spring_indices, self.spring_data,
                1.1,  # stretch_limit
                self.num_springs
            )

    def update(self, dt):
        self.particles.apply_gravity(Config.gravity)

        self.solve_constraints()
        
        self.particles.integrate(dt, Config.friction)

        if self.enable_self_collision:
            iters = getattr(Config, 'self_collision_iters', 2)
            self.solve_self_collision(iterations=iters)

        ground_collision_kernel[self.particles.blocks, self.threads_per_block](
            self.particles.pos, self.particles.vel, 0.0, self.particles.num
        )
    
    def solve_self_collision(self, iterations=2):
        if self.use_spatial_hash:
            for _ in range(iterations):
                self._update_spatial_hash()

                self_collision_spatial_hash_kernel[self.particles.blocks, self.threads_per_block](
                    self.particles.pos, self.particles.vel, self.particles.inv_mass,
                    self.adjacency_gpu, self.cell_indices, self.sorted_indices,
                    self.cell_offsets, self.cell_counts,
                    self.collision_thickness, self.cell_size, self.grid_size,
                    self.particles.num, self.num_cells
                )
        else:
            # 暴力版本（用于小网格）
            for _ in range(iterations):
                self_collision_kernel[self.particles.blocks, self.threads_per_block](
                    self.particles.pos, self.particles.vel, self.particles.inv_mass,
                    self.adjacency_gpu, self.collision_thickness, self.particles.num
                )

    def get_positions_cpu(self):
        return self.particles.to_numpy()
    
    def get_velocities_cpu(self):
        return self.particles.get_velocities()
    
    def set_positions_partial(self, indices, new_pos):
        self.particles.set_positions_partial(indices, new_pos)
    
    def set_velocities_partial(self, indices, new_vel):
        self.particles.set_velocities_partial(indices, new_vel)
    
    def solve_sphere_collision(self, sphere):
        threshold = sphere.radius + 0.06

        pos_before = self.particles.to_numpy()

        sphere_collision_kernel[self.particles.blocks, self.threads_per_block](
            self.particles.pos, self.particles.vel, self.particles.inv_mass,
            float(sphere.pos[0]), float(sphere.pos[1]), float(sphere.pos[2]),
            float(sphere.vel[0]), float(sphere.vel[1]), float(sphere.vel[2]),
            threshold, self.particles.num
        )

        pos_after = self.particles.to_numpy()
        delta_pos = pos_after - pos_before

        collision_mask = np.linalg.norm(delta_pos, axis=1) > 1e-6
        num_colliding = np.sum(collision_mask)
        
        if num_colliding > 0:
            diffs = pos_after[collision_mask] - sphere.pos
            dists = np.linalg.norm(diffs, axis=1)
            dists[dists < 1e-6] = 1e-6
            normals = diffs / dists[:, np.newaxis]

            avg_normal = np.mean(normals, axis=0)
            drag_force = -avg_normal * num_colliding * 0.5

            sphere.vel += drag_force * sphere.inv_mass * Config.dt

@cuda.jit(device=True)
def closest_point_on_triangle_device(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz):
    abx, aby, abz = bx - ax, by - ay, bz - az
    acx, acy, acz = cx - ax, cy - ay, cz - az
    apx, apy, apz = px - ax, py - ay, pz - az

    d1 = abx * apx + aby * apy + abz * apz
    d2 = acx * apx + acy * apy + acz * apz

    if d1 <= 0.0 and d2 <= 0.0:
        return ax, ay, az

    bpx, bpy, bpz = px - bx, py - by, pz - bz
    d3 = abx * bpx + aby * bpy + abz * bpz
    d4 = acx * bpx + acy * bpy + acz * bpz

    if d3 >= 0.0 and d4 <= d3:
        return bx, by, bz

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3 + 1e-10)
        return ax + abx * v, ay + aby * v, az + abz * v

    cpx, cpy, cpz = px - cx, py - cy, pz - cz
    d5 = abx * cpx + aby * cpy + abz * cpz
    d6 = acx * cpx + acy * cpy + acz * cpz

    if d6 >= 0.0 and d5 <= d6:
        return cx, cy, cz

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6 + 1e-10)
        return ax + acx * w, ay + acy * w, az + acz * w

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6) + 1e-10)
        return bx + (cx - bx) * w, by + (cy - by) * w, bz + (cz - bz) * w

    denom = 1.0 / (va + vb + vc + 1e-10)
    v = vb * denom
    w = vc * denom
    return ax + abx * v + acx * w, ay + aby * v + acy * w, az + abz * v + acz * w


@cuda.jit
def mesh_collision_kernel(particle_pos, particle_vel, particle_inv_mass,
                          mesh_verts, mesh_faces, tri_centers, tri_radii,
                          thickness, num_particles, num_triangles):
    i = cuda.grid(1)
    if i >= num_particles:
        return
    if particle_inv_mass[i] <= 0:
        return
    
    px = particle_pos[i, 0]
    py = particle_pos[i, 1]
    pz = particle_pos[i, 2]

    for tri_idx in range(num_triangles):
        cx = tri_centers[tri_idx, 0]
        cy = tri_centers[tri_idx, 1]
        cz = tri_centers[tri_idx, 2]
        radius = tri_radii[tri_idx] + thickness
        
        dx = px - cx
        dy = py - cy
        dz = pz - cz
        dist_sq = dx * dx + dy * dy + dz * dz
        
        if dist_sq >= radius * radius:
            continue

        f0 = mesh_faces[tri_idx, 0]
        f1 = mesh_faces[tri_idx, 1]
        f2 = mesh_faces[tri_idx, 2]
        
        ax, ay, az = mesh_verts[f0, 0], mesh_verts[f0, 1], mesh_verts[f0, 2]
        bx, by, bz = mesh_verts[f1, 0], mesh_verts[f1, 1], mesh_verts[f1, 2]
        cx, cy, cz = mesh_verts[f2, 0], mesh_verts[f2, 1], mesh_verts[f2, 2]

        cpx, cpy, cpz = closest_point_on_triangle_device(
            px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz
        )

        delta_x = px - cpx
        delta_y = py - cpy
        delta_z = pz - cpz
        dist = math.sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z)
        
        if dist < 1e-8:
            abx, aby, abz = bx - ax, by - ay, bz - az
            acx, acy, acz = cx - ax, cy - ay, cz - az
            nx = aby * acz - abz * acy
            ny = abz * acx - abx * acz
            nz = abx * acy - aby * acx
            norm_len = math.sqrt(nx * nx + ny * ny + nz * nz)
            if norm_len > 1e-8:
                nx /= norm_len
                ny /= norm_len
                nz /= norm_len
                particle_pos[i, 0] += nx * thickness * 0.5
                particle_pos[i, 1] += ny * thickness * 0.5
                particle_pos[i, 2] += nz * thickness * 0.5
            continue
        
        if dist >= thickness:
            continue

        nx = delta_x / dist
        ny = delta_y / dist
        nz = delta_z / dist
        
        correction = (thickness - dist + 1e-5) * 1.0
        particle_pos[i, 0] += nx * correction
        particle_pos[i, 1] += ny * correction
        particle_pos[i, 2] += nz * correction

        particle_vel[i, 0] *= 0.6
        particle_vel[i, 1] *= 0.6
        particle_vel[i, 2] *= 0.6


@cuda.jit
def mesh_collision_spatial_hash_kernel(particle_pos, particle_vel, particle_inv_mass,
                                       mesh_verts, mesh_faces, tri_centers, tri_radii,
                                       tri_cell_offsets, tri_cell_counts, tri_sorted_indices,
                                       thickness, cell_size, grid_size, num_particles, num_cells):
    i = cuda.grid(1)
    if i >= num_particles:
        return
    if particle_inv_mass[i] <= 0:
        return

    px = particle_pos[i, 0]
    py = particle_pos[i, 1]
    pz = particle_pos[i, 2]

    gx = int(math.floor(px / cell_size)) % grid_size
    gy = int(math.floor(py / cell_size)) % grid_size
    gz = int(math.floor(pz / cell_size)) % grid_size

    for dz in (-1, 0, 1):
        nz = gz + dz
        if nz < 0:
            nz += grid_size
        elif nz >= grid_size:
            nz -= grid_size
        for dy in (-1, 0, 1):
            ny = gy + dy
            if ny < 0:
                ny += grid_size
            elif ny >= grid_size:
                ny -= grid_size
            for dx in (-1, 0, 1):
                nx = gx + dx
                if nx < 0:
                    nx += grid_size
                elif nx >= grid_size:
                    nx -= grid_size

                cell_idx = nx + ny * grid_size + nz * grid_size * grid_size

                start = tri_cell_offsets[cell_idx]
                count = tri_cell_counts[cell_idx]

                for k in range(count):
                    tri_idx = tri_sorted_indices[start + k]

                    cx = tri_centers[tri_idx, 0]
                    cy = tri_centers[tri_idx, 1]
                    cz = tri_centers[tri_idx, 2]
                    radius = tri_radii[tri_idx] + thickness
                    dx0 = px - cx
                    dy0 = py - cy
                    dz0 = pz - cz
                    dist_sq = dx0 * dx0 + dy0 * dy0 + dz0 * dz0
                    if dist_sq >= radius * radius:
                        continue

                    f0 = mesh_faces[tri_idx, 0]
                    f1 = mesh_faces[tri_idx, 1]
                    f2 = mesh_faces[tri_idx, 2]

                    ax, ay, az = mesh_verts[f0, 0], mesh_verts[f0, 1], mesh_verts[f0, 2]
                    bx, by, bz = mesh_verts[f1, 0], mesh_verts[f1, 1], mesh_verts[f1, 2]
                    cx2, cy2, cz2 = mesh_verts[f2, 0], mesh_verts[f2, 1], mesh_verts[f2, 2]

                    cpx, cpy, cpz = closest_point_on_triangle_device(
                        px, py, pz, ax, ay, az, bx, by, bz, cx2, cy2, cz2
                    )

                    delta_x = px - cpx
                    delta_y = py - cpy
                    delta_z = pz - cpz
                    dist = math.sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z)

                    if dist < 1e-8:
                        abx, aby, abz = bx - ax, by - ay, bz - az
                        acx, acy, acz = cx2 - ax, cy2 - ay, cz2 - az
                        nxn = aby * acz - abz * acy
                        nyn = abz * acx - abx * acz
                        nzn = abx * acy - aby * acx
                        norm_len = math.sqrt(nxn * nxn + nyn * nyn + nzn * nzn)
                        if norm_len > 1e-8:
                            nxn /= norm_len
                            nyn /= norm_len
                            nzn /= norm_len
                            particle_pos[i, 0] += nxn * thickness * 0.5
                            particle_pos[i, 1] += nyn * thickness * 0.5
                            particle_pos[i, 2] += nzn * thickness * 0.5
                        continue

                    if dist >= thickness:
                        continue

                    nxn = delta_x / dist
                    nyn = delta_y / dist
                    nzn = delta_z / dist
                    correction = (thickness - dist + 1e-5) * 1.0
                    particle_pos[i, 0] += nxn * correction
                    particle_pos[i, 1] += nyn * correction
                    particle_pos[i, 2] += nzn * correction
                    particle_vel[i, 0] *= 0.6
                    particle_vel[i, 1] *= 0.6
                    particle_vel[i, 2] *= 0.6


class MeshColliderCUDA:
    def __init__(self, obj_path, scale=1.0, offset=(0, 0, 0), rotation=(0, 0, 0), thickness=0.05, mass=1.0):
        verts, faces = load_obj_mesh(obj_path)
        
        self.verts_original = (verts * scale).astype(np.float32)
        self.faces_cpu = np.array(faces, dtype=np.int32)
        self.num_triangles = len(faces)
        self.thickness = thickness
        
        self.scale = scale
        self.offset_init = np.array(offset, dtype=np.float32)
        self.rotation_init = np.array(rotation, dtype=np.float32)
        
        self.mass = mass
        self.inv_mass = 1.0 / mass if mass > 0 else 0.0
        self.damping = 0.999998
        
        self.pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.rot = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.angular_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        self.force = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.torque = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        self._update_verts()

        self.faces_gpu = cuda.to_device(self.faces_cpu)
        self.verts_gpu = cuda.to_device(self.verts.astype(np.float32))
        self.centers_gpu = cuda.to_device(self.centers.astype(np.float32))
        self.radii_gpu = cuda.to_device(self.radii.astype(np.float32))

        self.threads_per_block = 256

        self.tri_use_spatial_hash = self.num_triangles > 64
        if self.tri_use_spatial_hash:
            self._init_triangle_spatial_hash()
        
        print(f"MeshColliderCUDA initialized: {len(verts)} verts, {self.num_triangles} triangles")
    
    def _update_verts(self):
        total_rot = self.rotation_init + self.rot
        rx, ry, rz = np.deg2rad(total_rot)
        
        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]], dtype=np.float32)
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]], dtype=np.float32)
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]], dtype=np.float32)
        R = Rz @ Ry @ Rx
        
        verts = self.verts_original @ R.T
        self.verts = verts + self.offset_init + self.pos

        tris = self.verts[self.faces_cpu]
        self.centers = np.mean(tris, axis=1).astype(np.float32)
        self.radii = np.linalg.norm(tris - self.centers[:, np.newaxis, :], axis=2).max(axis=1).astype(np.float32)
    
    def _sync_to_gpu(self):
        self.verts_gpu = cuda.to_device(self.verts.astype(np.float32))
        self.centers_gpu = cuda.to_device(self.centers.astype(np.float32))
        self.radii_gpu = cuda.to_device(self.radii.astype(np.float32))

    def _init_triangle_spatial_hash(self):
        self.tri_cell_size = self.thickness * 1.5
        self.tri_grid_size = 64
        self.tri_num_cells = self.tri_grid_size ** 3

        self.tri_cell_indices = cuda.to_device(np.zeros(self.num_triangles, dtype=np.int32))
        self.tri_sorted_indices = cuda.to_device(np.zeros(self.num_triangles, dtype=np.int32))
        self.tri_cell_counts = cuda.to_device(np.zeros(self.tri_num_cells, dtype=np.int32))
        self.tri_cell_offsets = cuda.to_device(np.zeros(self.tri_num_cells, dtype=np.int32))
        self.tri_cell_counts_temp = cuda.to_device(np.zeros(self.tri_num_cells, dtype=np.int32))

        print(f"Triangle spatial hash initialized: grid {self.tri_grid_size}^3, cell_size={self.tri_cell_size:.3f}")

    def _update_triangle_spatial_hash(self):
        tpb = self.threads_per_block
        cell_blocks = (self.tri_num_cells + tpb - 1) // tpb
        clear_array_kernel[cell_blocks, tpb](self.tri_cell_counts, self.tri_num_cells)
        clear_array_kernel[cell_blocks, tpb](self.tri_cell_counts_temp, self.tri_num_cells)

        blocks = (self.num_triangles + tpb - 1) // tpb
        compute_cell_indices_kernel[blocks, tpb](
            self.centers_gpu, self.tri_cell_indices, self.tri_cell_size, self.tri_grid_size, self.num_triangles
        )

        count_particles_per_cell_kernel[blocks, tpb](
            self.tri_cell_indices, self.tri_cell_counts, self.num_triangles
        )

        compute_cell_offsets_cpu(self.tri_cell_counts, self.tri_cell_offsets, self.tri_num_cells)
        sort_particles_kernel[blocks, tpb](
            self.tri_cell_indices, self.tri_cell_offsets, self.tri_cell_counts_temp,
            self.tri_sorted_indices, self.num_triangles
        )
        cuda.synchronize()
    
    def update(self, dt, gravity=None):
        if gravity is None:
            gravity = np.array([0.0, -9.8, 0.0], dtype=np.float32)
        
        if self.inv_mass > 0:
            self.force += self.mass * gravity
            self.vel += self.force * self.inv_mass * dt
            self.vel *= self.damping
            self.pos += self.vel * dt
            
            inertia_approx = self.mass * 0.1
            angular_accel = self.torque / (inertia_approx + 1e-6)
            self.angular_vel += np.rad2deg(angular_accel) * dt
            self.angular_vel *= self.damping
            self.rot += self.angular_vel * dt

        self._update_verts()
        self._sync_to_gpu()

        self.force.fill(0.0)
        self.torque.fill(0.0)
    
    def collide(self, cloth_or_particles, iterations=2):
        if hasattr(cloth_or_particles, 'particles'):
            particles = cloth_or_particles.particles
        else:
            particles = cloth_or_particles
        
        num_particles = particles.num
        blocks = (num_particles + self.threads_per_block - 1) // self.threads_per_block

        pos_before = particles.pos.copy_to_host()
        
        for _ in range(iterations):
            mult = getattr(Config, 'mesh_cloth_collision_multiplier', 1.5)
            cloth_th = getattr(Config, 'self_collision_thickness', 0.05)
            effective_thickness = max(self.thickness, cloth_th * mult)

            if hasattr(self, 'tri_use_spatial_hash') and self.tri_use_spatial_hash:
                self._update_triangle_spatial_hash()
                mesh_collision_spatial_hash_kernel[blocks, self.threads_per_block](
                    particles.pos, particles.vel, particles.inv_mass,
                    self.verts_gpu, self.faces_gpu, self.centers_gpu, self.radii_gpu,
                    self.tri_cell_offsets, self.tri_cell_counts, self.tri_sorted_indices,
                    float(effective_thickness), float(self.tri_cell_size), int(self.tri_grid_size),
                    num_particles, int(self.tri_num_cells)
                )
            else:
                mesh_collision_kernel[blocks, self.threads_per_block](
                    particles.pos, particles.vel, particles.inv_mass,
                    self.verts_gpu, self.faces_gpu, self.centers_gpu, self.radii_gpu,
                    float(effective_thickness), num_particles, self.num_triangles
                )

        if self.inv_mass > 0:
            pos_after = particles.pos.copy_to_host()
            delta_pos = pos_after - pos_before
            collision_mask = np.linalg.norm(delta_pos, axis=1) > 1e-6
            num_colliding = np.sum(collision_mask)
            if num_colliding > 0:
                avg_delta = np.mean(delta_pos[collision_mask], axis=0)
                impulse_strength = 0.05
                impulse = avg_delta * num_colliding * impulse_strength
                self.vel -= impulse * self.inv_mass


if __name__ == "__main__":
    print("Testing CUDA cloth simulation...")
    print(f"CUDA available: {cuda.is_available()}")
    
    if cuda.is_available():
        print(f"GPU: {cuda.get_current_device().name}")

