import numpy as np
from collections import defaultdict

class SpatialHash:
    def __init__(self, cell_size):
        self.cell_size = float(cell_size)
        self.grid = {}
        self.positions = None
        self.num_particles = 0

    def build(self, positions):
        self.grid = {}
        self.positions = positions
        self.num_particles = len(positions)
        
        temp_grid = defaultdict(list)
        
        keys = np.floor(positions / self.cell_size).astype(np.int32)
        
        for i, key in enumerate(keys):
            k = tuple(key)
            temp_grid[k].append(i)
        
        self.grid = dict(temp_grid)

    def query(self, pos):
        # 返回周围 3x3x3 格子里的所有粒子索引
        center = np.floor(np.asarray(pos) / self.cell_size).astype(np.int32)
        indices = []
        
        cx, cy, cz = center
        neighbor_keys = [
            (cx+dx, cy+dy, cz+dz) 
            for dx in (-1, 0, 1) 
            for dy in (-1, 0, 1) 
            for dz in (-1, 0, 1)
        ]
        
        for key in neighbor_keys:
            if key in self.grid:
                indices.extend(self.grid[key])
        
        return indices
    
    def query_batch(self, positions):
        # 批量查询 返回每个位置的邻近粒子索引
        results = []
        keys_batch = np.floor(positions / self.cell_size).astype(np.int32)
        
        for center in keys_batch:
            indices = []
            cx, cy, cz = center
            
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        key = (cx+dx, cy+dy, cz+dz)
                        if key in self.grid:
                            indices.extend(self.grid[key])
            
            results.append(indices)
        
        return results
    
    def query_radius(self, pos, radius):
        # 半径范围查询 查找距离 pos 在 radius 内的粒子
        search_range = int(np.ceil(radius / self.cell_size))
        center = np.floor(np.asarray(pos) / self.cell_size).astype(np.int32)
        
        candidates = []
        cx, cy, cz = center
        
        for dx in range(-search_range, search_range + 1):
            for dy in range(-search_range, search_range + 1):
                for dz in range(-search_range, search_range + 1):
                    key = (cx+dx, cy+dy, cz+dz)
                    if key in self.grid:
                        candidates.extend(self.grid[key])

        if self.positions is not None:
            pos_arr = np.asarray(pos)
            result = []
            radius_sq = radius * radius
            
            for idx in candidates:
                dist_sq = np.sum((self.positions[idx] - pos_arr) ** 2)
                if dist_sq <= radius_sq:
                    result.append(idx)
            
            return result
        
        return candidates
    
    def query_aabb(self, min_pos, max_pos):
        # 轴对齐包围盒查询 查找在 AABB 内的粒子
        min_key = np.floor(np.asarray(min_pos) / self.cell_size).astype(np.int32)
        max_key = np.floor(np.asarray(max_pos) / self.cell_size).astype(np.int32)
        
        candidates = []
        
        for cx in range(min_key[0], max_key[0] + 1):
            for cy in range(min_key[1], max_key[1] + 1):
                for cz in range(min_key[2], max_key[2] + 1):
                    key = (cx, cy, cz)
                    if key in self.grid:
                        candidates.extend(self.grid[key])

        if self.positions is not None:
            result = []
            min_arr = np.asarray(min_pos)
            max_arr = np.asarray(max_pos)
            
            for idx in candidates:
                p = self.positions[idx]
                if np.all(p >= min_arr) and np.all(p <= max_arr):
                    result.append(idx)
            
            return result
        
        return candidates
    
    def get_cell_info(self):
        if not self.grid:
            return {
                'num_cells': 0,
                'num_particles': 0,
                'avg_particles_per_cell': 0,
                'max_particles_per_cell': 0,
                'fill_ratio': 0
            }
        
        particle_counts = [len(v) for v in self.grid.values()]
        
        return {
            'num_cells': len(self.grid),
            'num_particles': self.num_particles,
            'avg_particles_per_cell': np.mean(particle_counts),
            'max_particles_per_cell': np.max(particle_counts),
            'min_particles_per_cell': np.min(particle_counts),
            'fill_ratio': len(self.grid) / self.num_particles if self.num_particles > 0 else 0
        }