import numpy as np

# 导出布料和可选的球体、固体信息到 OBJ 文件
def export_frame(cloth, sphere, frame_id, collider=None):
    filename = f"output/frame_{frame_id:04d}.obj"
    with open(filename, 'w') as f:
        f.write(f"# Simulation Frame {frame_id}\n")
        
        for p in cloth.particles.pos:
            f.write(f"v {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
            
        if hasattr(cloth, 'faces'):
            for face in cloth.faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
                
        elif hasattr(cloth, 'rows') and hasattr(cloth, 'cols'):
            rows, cols = cloth.rows, cloth.cols
            for r in range(rows - 1):
                for c in range(cols - 1):
                    idx = r * cols + c
                    
                    v1 = idx + 1
                    v2 = idx + 1 + 1
                    v3 = idx + cols + 1
                    v4 = idx + cols + 1 + 1
                    
                    f.write(f"f {v1} {v3} {v2}\n")
                    f.write(f"f {v2} {v3} {v4}\n")

    log_msg = f"Frame {frame_id} exported."
    if collider is not None:
        log_msg += f" Collider: pos={collider.pos+collider.offset_init}, rot={collider.rot}"
    else:
        log_msg += f" Sphere: pos={sphere.pos}"
    print(log_msg)

def load_obj_mesh(obj_path):
    vertices = []
    faces = []
    
    with open(obj_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if parts[0] == 'v':
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                face = []
                for part in parts[1:]:
                    vertex_index = int(part.split('/')[0]) - 1
                    face.append(vertex_index)
                faces.append(face)
    
    return np.array(vertices, dtype=np.float64), faces