import numpy as np
from config import Config
from utils import load_obj_mesh
from spatial_hash import SpatialHash


def _closest_point_on_triangle(p, a, b, c):
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a

    bp = p - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab

    cp = p - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b)

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return a + ab * v + ac * w

class ParticleSystem:
    def __init__(self, num_particles):
        self.num = num_particles
        self.pos = np.zeros((num_particles, 3))
        self.vel = np.zeros((num_particles, 3))
        self.inv_mass = np.ones(num_particles)
        self.forces = np.zeros((num_particles, 3))

    def apply_gravity(self):
        non_fixed = self.inv_mass > 0
        
        if np.any(non_fixed):
            mass = 1.0 / self.inv_mass[non_fixed]            
            gravity_force = mass[:, np.newaxis] * Config.gravity
            self.forces[non_fixed] += gravity_force

    def integrate(self, dt):
        self.forces = np.nan_to_num(self.forces, nan=0.0, posinf=0.0, neginf=0.0)

        acc = self.forces * self.inv_mass[:, np.newaxis]
        self.vel += acc * dt
        self.vel *= Config.friction
        
        self.vel = np.nan_to_num(self.vel, nan=0.0, posinf=0.0, neginf=0.0)

        max_vel = 30.0
        
        self.vel = np.clip(self.vel, -1e5, 1e5)
        
        speed_sq = np.sum(self.vel * self.vel, axis=1)
        
        dangerous = speed_sq > (max_vel * max_vel)
        
        if np.any(dangerous):
            speed = np.sqrt(speed_sq[dangerous])
            scale = max_vel / (speed + 1e-8)
            self.vel[dangerous] *= scale[:, np.newaxis]

        self.pos += self.vel * dt
        self.forces[:] = 0.0


class MeshCollider:
    def __init__(self, obj_path, scale=1.0, offset=(0, 0, 0), rotation=(0, 0, 0), thickness=0.05, mass=1.0):
        verts, faces = load_obj_mesh(obj_path)
        
        self.verts_original = (verts * scale).astype(np.float64)
        self.faces = np.array(faces, dtype=np.int32)
        self.thickness = thickness
        
        self.scale = scale
        self.offset_init = np.array(offset, dtype=np.float64)
        self.rotation_init = np.array(rotation, dtype=np.float64)
        
        self.mass = mass
        self.inv_mass = 1.0 / mass if mass > 0 else 0.0
        self.damping = 0.9999
        
        self.pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.rot = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.angular_vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        
        self.force = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.torque = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        
        self.offset_init = np.array(offset, dtype=np.float64)
        self._update_verts()
        
        self.spatial_hash = SpatialHash(cell_size=thickness * 2.5)
        self._build_spatial_hash()
    
    def _update_verts(self):
        total_rot = self.rotation_init + self.rot
        rx, ry, rz = np.deg2rad(total_rot)
        
        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        
        verts = self.verts_original @ R.T
        self.verts = verts + self.offset_init + self.pos
        
        tris = self.verts[self.faces]
        self.centers = np.mean(tris, axis=1)
        self.radii = np.linalg.norm(tris - self.centers[:, np.newaxis, :], axis=2).max(axis=1)
    
    def _build_spatial_hash(self):
        self.spatial_hash.build(self.centers)
    
    def update(self, dt, gravity=None):
        if gravity is None:
            gravity = np.array([0.0, -9.8, 0.0], dtype=np.float64)
        
        if self.inv_mass > 0:
            self.force += self.mass * gravity
            
            self.vel += self.force * self.inv_mass * dt
            self.vel *= self.damping
            
            self.pos += self.vel * dt
            
            inertia_approx = self.mass * 0.1  # 粗略估计
            angular_accel = self.torque / (inertia_approx + 1e-6)
            self.angular_vel += np.rad2deg(angular_accel) * dt
            self.angular_vel *= self.damping

            self.rot += self.angular_vel * dt

    def collide(self, particles, iterations=2):
        self._update_verts()
        self._build_spatial_hash()
        
        self.force.fill(0.0)
        self.torque.fill(0.0)
        
        pos = particles.pos
        vel = particles.vel
        inv_mass = particles.inv_mass
        t = self.thickness

        for _ in range(iterations):
            for pi in range(len(pos)):
                if inv_mass[pi] <= 0:
                    continue
                
                p = pos[pi]
                
                nearby_tri_indices = self.spatial_hash.query_radius(p, radius=t * 3.0)
                
                if not nearby_tri_indices:
                    continue

                for tri_idx in nearby_tri_indices:
                    face = self.faces[tri_idx]
                    a, b, c = self.verts[face[0]], self.verts[face[1]], self.verts[face[2]]
                    center = self.centers[tri_idx]
                    radius = self.radii[tri_idx] + t

                    diff = p - center
                    dist_sq = np.sum(diff * diff)
                    if dist_sq >= radius * radius:
                        continue

                    cp = _closest_point_on_triangle(p, a, b, c)
                    delta = p - cp
                    dist = np.linalg.norm(delta)
                    
                    if dist < 1e-8:
                        ab = b - a
                        ac = c - a
                        normal = np.cross(ab, ac)
                        norm_len = np.linalg.norm(normal)
                        if norm_len > 1e-8:
                            n = normal / norm_len
                            pos[pi] += n * t * 0.5
                            particle_mass = 1.0 / inv_mass[pi] if inv_mass[pi] > 0 else 0.001
                            impulse = n * particle_mass * 0.1
                            self.force += impulse
                        continue
                    
                    if dist >= t:
                        continue
                    
                    n = delta / dist
                    correction = (t - dist + 1e-5) * n * 1.5
                    pos[pi] = p + correction
                    
                    particle_mass = 1.0 / inv_mass[pi] if inv_mass[pi] > 0 else 0.001
                    impulse = n * particle_mass * (t - dist) * 0.5
                    self.force += impulse
                    
                    r = cp - self.pos
                    torque = np.cross(r, impulse)
                    self.torque += torque
                    
                    vel[pi] *= 0.2

class SphereBody:
    def __init__(self, pos, vel, radius, mass):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.radius = radius
        self.mass = mass
        self.inv_mass = 1.0 / mass if mass > 0 else 0.0

    def update(self, dt):
        if self.inv_mass == 0: return
        force = Config.gravity * self.mass
        self.vel += (force / self.mass) * dt
        self.vel *= Config.friction
        self.pos += self.vel * dt

        if self.pos[1] < self.radius:
            self.pos[1] = self.radius
            self.vel[1] = -self.vel[1] * 0.8

class Cloth:
    def __init__(self):
        rows, cols = Config.cloth_res
        w, h = Config.cloth_size
        self.rows = rows
        self.cols = cols
        num_particles = rows * cols
        self.particles = ParticleSystem(num_particles)
        
        x = np.linspace(-w/2, w/2, cols)
        z = np.linspace(-h/2, h/2, rows)
        xx, zz = np.meshgrid(x, z)
        self.particles.pos[:, 0] = xx.flatten()
        self.particles.pos[:, 1] = 6.0 
        self.particles.pos[:, 2] = zz.flatten()

        self.particles.vel[:] = Config.cloth_initial_velocity
        
        self.particles.inv_mass[0] = 0 
        self.particles.inv_mass[cols-1] = 0
        self.particles.inv_mass[(rows-1)*cols] = 0
        self.particles.inv_mass[rows*cols-1] = 0

        self.springs = []
        def add_spring(i, j):
            if i < num_particles and j < num_particles:
                p1 = self.particles.pos[i]
                p2 = self.particles.pos[j]
                rest_len = np.linalg.norm(p1 - p2)
                self.springs.append((i, j, rest_len))

        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                if c < cols - 1: add_spring(idx, idx + 1)
                if r < rows - 1: add_spring(idx, idx + cols)
        self.spatial_hash = SpatialHash(cell_size=0.1)

    def solve_constraints(self):
        for (p1_idx, p2_idx, rest_len) in self.springs:
            p1 = self.particles.pos[p1_idx]
            p2 = self.particles.pos[p2_idx]
            v1 = self.particles.vel[p1_idx]
            v2 = self.particles.vel[p2_idx]
            
            delta = p1 - p2
            dist = np.linalg.norm(delta)
            if dist == 0: continue
            
            force_mag = Config.k_stiffness * (dist - rest_len)
            force_vec = (delta / dist) * force_mag
            
            rel_vel = v1 - v2
            damping = 1.0 * np.dot(rel_vel, delta/dist) * (delta/dist)
            
            total_force = -(force_vec + damping)
            self.particles.forces[p1_idx] += total_force
            self.particles.forces[p2_idx] -= total_force
        '''
        stretch_limit = 1.1

        for _ in range(Config.stretch_limit_iters):
            for (p1_idx, p2_idx, rest_len) in self.springs:
                p1 = self.particles.pos[p1_idx]
                p2 = self.particles.pos[p2_idx]
                
                delta = p1 - p2
                dist = np.linalg.norm(delta)
                
                if dist > rest_len * stretch_limit:
                    diff = (dist - rest_len * stretch_limit) / dist
                    correction = delta * 0.5 * diff
                    
                    if self.particles.inv_mass[p1_idx] > 0:
                        self.particles.pos[p1_idx] -= correction
                    if self.particles.inv_mass[p2_idx] > 0:
                        self.particles.pos[p2_idx] += correction
        '''

    def update(self, dt):
        self.particles.apply_gravity()
        self.solve_constraints()
        self.particles.integrate(dt)
        
        mask = self.particles.pos[:, 1] < 0
        self.particles.pos[mask, 1] = 0
        self.particles.vel[mask, 1] *= -0.1

class MeshCloth:
    def __init__(self, obj_path, scale=1.0, offset=(0, 0, 0), rotation=(0, 0, 0)):
        print(f"Loading mesh from {obj_path}...")
        raw_verts, raw_faces = load_obj_mesh(obj_path)

        self.faces = np.array(raw_faces, dtype=np.int32)

        num_particles = len(raw_verts)
        print(f"Mesh loaded: {num_particles} vertices, {len(raw_faces)} faces.")
        
        self.particles = ParticleSystem(num_particles)
        
        self.particles.pos = raw_verts * scale
        
        rx, ry, rz = np.deg2rad(rotation)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(rx), -np.sin(rx)],
                       [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                       [0, 1, 0],
                       [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                       [np.sin(rz), np.cos(rz), 0],
                       [0, 0, 1]])
        R = Rz @ Ry @ Rx
        self.particles.pos = self.particles.pos @ R.T
        
        self.particles.pos += np.array(offset)

        self.particles.vel[:] = Config.cloth_initial_velocity
        
        total_mass = 2.0
        particle_mass = total_mass / num_particles
        self.particles.inv_mass[:] = 1.0 / particle_mass
        
        #max_y = np.max(self.particles.pos[:, 1])
        #top_mask = self.particles.pos[:, 1] > (max_y - 0.05) 
        #self.particles.inv_mass[top_mask] = 0
        
        pos = self.particles.pos
        '''
        search_dir = np.array([-1.0, 1.0, 1.0]) 
        scores = np.dot(pos, search_dir)
        corner_idx = np.argmax(scores)
        self.particles.inv_mass[corner_idx] = 0
        print(f"Pinned particle {corner_idx} at {pos[corner_idx]}")

        search_dir = np.array([-1.0, 1.0, -1.0]) 
        scores = np.dot(pos, search_dir)
        corner_idx = np.argmax(scores)
        self.particles.inv_mass[corner_idx] = 0
        print(f"Pinned particle {corner_idx} at {pos[corner_idx]}")

        search_dir = np.array([1.0, 1.0, -1.0]) 
        scores = np.dot(pos, search_dir)
        corner_idx = np.argmax(scores)
        self.particles.inv_mass[corner_idx] = 0
        print(f"Pinned particle {corner_idx} at {pos[corner_idx]}")

        search_dir = np.array([1.0, 1.0, 1.0]) 
        scores = np.dot(pos, search_dir)
        corner_idx = np.argmax(scores)
        self.particles.inv_mass[corner_idx] = 0
        print(f"Pinned particle {corner_idx} at {pos[corner_idx]}")
        '''

        spring_indices = []
        spring_data = [] 
        existing_links = set()
        
        def add_link(i, j, stiffness_ratio):
            if i == j: return
            if i > j: i, j = j, i
            if (i, j) in existing_links: return
            
            p1 = self.particles.pos[i]
            p2 = self.particles.pos[j]
            rest_len = np.linalg.norm(p1 - p2)
            
            if rest_len < 0.0001: return

            k = Config.k_stiffness * stiffness_ratio
            spring_indices.append([i, j])
            spring_data.append([rest_len, k])
            existing_links.add((i, j))

        print("Building Structural Springs (Edges)...")
        # 结构弹簧 (基于 Mesh 边)
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
        
        # 弯曲弹簧 (对顶点)
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
                    add_link(opp1[0], opp2[0], 0.3)  # 刚度30%
        
        self.spring_indices = np.array(spring_indices, dtype=np.int32)
        self.spring_data = np.array(spring_data, dtype=np.float64)
        
        print(f"Springs built: {len(spring_indices)} constraints.")
        
        self.adjacency_set = existing_links.copy()
        print(f"Adjacency set built: {len(self.adjacency_set)} edges excluded from self-collision.")
        
        self.spatial_hash = SpatialHash(cell_size=Config.self_collision_thickness * 2.5)

    def apply_aerodynamics(self, wind_velocity):
        f0 = self.faces[:, 0]
        f1 = self.faces[:, 1]
        f2 = self.faces[:, 2]
        
        pos = self.particles.pos
        vel = self.particles.vel
        
        p0 = pos[f0]
        p1 = pos[f1]
        p2 = pos[f2]
        
        v0 = vel[f0]
        v1 = vel[f1]
        v2 = vel[f2]

        v_tri = (v0 + v1 + v2) / 3.0

        v_rel = wind_velocity - v_tri
        
        edge1 = p1 - p0
        edge2 = p2 - p0
        cross = np.cross(edge1, edge2)
        
        cross_dot_vel = np.sum(cross * v_rel, axis=1)
        
        drag_coeff = 0.05 
              
        force_mag = drag_coeff * cross_dot_vel
        face_force = cross * force_mag[:, np.newaxis]
        
        vertex_force = face_force / 3.0
        
        np.add.at(self.particles.forces, f0, vertex_force)
        np.add.at(self.particles.forces, f1, vertex_force)
        np.add.at(self.particles.forces, f2, vertex_force)

    def solve_constraints(self):
        p1_idx = self.spring_indices[:, 0]
        p2_idx = self.spring_indices[:, 1]
        
        rest_lens = self.spring_data[:, 0]
        ks = self.spring_data[:, 1]
        
        pos1 = self.particles.pos[p1_idx]
        pos2 = self.particles.pos[p2_idx]
        
        delta = pos1 - pos2
        dist = np.linalg.norm(delta, axis=1)
        
        dist[dist < 1e-6] = 1e-6
        
        direction = delta / dist[:, np.newaxis]
        
        raw_forces = ks * (dist - rest_lens)

        # 拉伸比压缩恢复力更强
        force_mags = np.where(
            raw_forces > 0,
            raw_forces * Config.tension_scale,
            raw_forces * Config.support_scale,
        )
        
        vel1 = self.particles.vel[p1_idx]
        vel2 = self.particles.vel[p2_idx]
        rel_vel = vel1 - vel2
        
        damping_factor = 2.0
        damping_term = np.sum(rel_vel * direction, axis=1) * damping_factor
        
        total_force_mag = force_mags + damping_term
        
        total_force = direction * total_force_mag[:, np.newaxis]
        
        np.add.at(self.particles.forces, p2_idx, total_force)
        np.add.at(self.particles.forces, p1_idx, -total_force)

        stretch_limit = 1.1

        for _ in range(Config.stretch_limit_iters):
            pos1_corr = self.particles.pos[p1_idx]
            pos2_corr = self.particles.pos[p2_idx]
            delta_corr = pos1_corr - pos2_corr
            dist_corr = np.linalg.norm(delta_corr, axis=1)
            dist_corr[dist_corr < 1e-6] = 1e-6

            over_stretched_mask = dist_corr > (rest_lens * stretch_limit)
            
            if np.any(over_stretched_mask):
                idx1_bad = p1_idx[over_stretched_mask]
                idx2_bad = p2_idx[over_stretched_mask]
                
                delta_bad = delta_corr[over_stretched_mask]
                dist_bad = dist_corr[over_stretched_mask]
                max_len_bad = rest_lens[over_stretched_mask] * stretch_limit
                
                correction_scalar = (dist_bad - max_len_bad) / dist_bad
                correction_vec = delta_bad * 0.5 * correction_scalar[:, np.newaxis]
                
                w1 = self.particles.inv_mass[idx1_bad]
                w2 = self.particles.inv_mass[idx2_bad]
                total_w = w1 + w2

                total_w[total_w == 0] = 1.0
                
                scale1 = w1 / total_w
                scale2 = w2 / total_w
                
                np.add.at(self.particles.pos, idx1_bad, -correction_vec * scale1[:, np.newaxis])
                np.add.at(self.particles.pos, idx2_bad, correction_vec * scale2[:, np.newaxis])

    def solve_self_collision(self):
        pos = self.particles.pos

        self.spatial_hash.build(pos)
        
        thickness = Config.self_collision_thickness
        min_dist = thickness * 2.0
        min_dist_sq = min_dist * min_dist
        
        num_particles = len(pos)
        checked_pairs = set()
        
        for p_idx_1 in range(num_particles):
            candidates = self.spatial_hash.query_radius(pos[p_idx_1], min_dist)
            
            for p_idx_2 in candidates:
                if p_idx_2 == p_idx_1:
                    continue
                
                pair = (min(p_idx_1, p_idx_2), max(p_idx_1, p_idx_2))
                if pair in checked_pairs:
                    continue
                checked_pairs.add(pair)
                
                if pair in self.adjacency_set:
                    continue
                
                delta = pos[p_idx_1] - pos[p_idx_2]
                dist_sq = np.dot(delta, delta)
                    
                if dist_sq < min_dist_sq and dist_sq > 1e-8:
                    dist = np.sqrt(dist_sq)
                    
                    correction_factor = 1.5
                    correction = (min_dist - dist) * correction_factor * (delta / dist)
                    
                    w1 = self.particles.inv_mass[p_idx_1]
                    w2 = self.particles.inv_mass[p_idx_2]
                    total_w = w1 + w2
                    
                    if total_w > 1e-8:
                        if w1 > 0: 
                            self.particles.pos[p_idx_1] += correction * (w1 / total_w)
                        if w2 > 0: 
                            self.particles.pos[p_idx_2] -= correction * (w2 / total_w)
                    
                    v1 = self.particles.vel[p_idx_1]
                    v2 = self.particles.vel[p_idx_2]
                    rel_vel = v1 - v2
                    
                    normal = delta / dist
                    v_normal = np.dot(rel_vel, normal)
                    
                    if v_normal < 0:
                        damping = 0.8
                        impulse = v_normal * normal * damping
                        if w1 > 0: 
                            self.particles.vel[p_idx_1] -= impulse
                        if w2 > 0: 
                            self.particles.vel[p_idx_2] += impulse
                    
                    friction_impulse = rel_vel * 0.3
                    if w1 > 0: 
                        self.particles.vel[p_idx_1] -= friction_impulse
                    if w2 > 0: 
                        self.particles.vel[p_idx_2] += friction_impulse

    def update(self, dt):
        self.particles.apply_gravity()
        
        import time
        t = time.time()
        
        base_wind = np.array([0.0, 0.0, -100.0])
        wind_strength = 2.0 + np.sin(t * 4.0) * 0.5
        current_wind = base_wind * wind_strength
        
        self.apply_aerodynamics(current_wind)
        

        self.solve_constraints()
        
        for _ in range(Config.self_collision_iters):
            self.solve_self_collision()

        self.particles.integrate(dt)
        
        ground_y = 0.0
        mask = self.particles.pos[:, 1] < ground_y
        
        if np.any(mask):
            self.particles.pos[mask, 1] = ground_y
            
            self.particles.vel[mask, 1] *= -0.1 
            
            ground_friction = 0.6
            self.particles.vel[mask, 0] *= ground_friction
            self.particles.vel[mask, 2] *= ground_friction