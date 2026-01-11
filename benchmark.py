import numpy as np
import time
import matplotlib.pyplot as plt
from config import Config

plt.rcParams['axes.unicode_minus'] = False

try:
    from numba import cuda
    CUDA_AVAILABLE = cuda.is_available()
    if CUDA_AVAILABLE:
        from objects_cuda import MeshClothCUDA, ParticleSystemCUDA
        from objects_cuda import (
            self_collision_kernel, 
            self_collision_spatial_hash_kernel,
            compute_cell_indices_kernel,
            count_particles_per_cell_kernel,
            compute_cell_offsets_cpu,
            sort_particles_kernel,
            clear_array_kernel
        )
        print(f"CUDA Available: {cuda.get_current_device().name}")
except ImportError:
    CUDA_AVAILABLE = False
    print("Numba CUDA not available")

from objects import MeshCloth

def benchmark_cpu_no_selfcoll(cloth, dt, warmup=3, iterations=20):
    """测试CPU版本性能 (不包含自碰撞)"""
    # 预热
    for _ in range(warmup):
        cloth.particles.forces[:] = 0
        cloth.particles.apply_gravity()
        cloth.solve_constraints()
        cloth.particles.integrate(dt)
    
    start = time.perf_counter()
    for _ in range(iterations):
        cloth.particles.forces[:] = 0
        cloth.particles.apply_gravity()
        cloth.solve_constraints()
        cloth.particles.integrate(dt)
    elapsed = time.perf_counter() - start
    
    avg_ms = elapsed / iterations * 1000
    fps = iterations / elapsed
    
    return avg_ms, fps

def benchmark_cuda(cloth_cuda, dt, warmup=10, iterations=100):
    """测试CUDA版本性能"""
    for _ in range(warmup):
        cloth_cuda.update(dt)
    cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iterations):
        cloth_cuda.update(dt)
    cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    avg_ms = elapsed / iterations * 1000
    fps = iterations / elapsed
    
    return avg_ms, fps


def test_scaling():
    """测试不同规模下的加速比 - 两个独立实验"""
    from mesh_generator import export_obj
    import os
    
    # 实验1使用较小规模，实验2使用更大规模以展示空间哈希优势
    sizes_exp1 = [20, 40, 60, 80, 100]  # 网格分辨率
    sizes_exp2 = [50, 100, 150, 200, 250]
    
    # 实验1: CUDA vs CPU (不含自碰撞)
    print("\n" + "="*70)
    print("EXPERIMENT 1: CUDA vs CPU (Without Self-Collision)")
    print("="*70)
    print("公平对比：两者都只计算弹簧力、重力、积分")
    
    results_exp1 = []
    
    for size in sizes_exp1:
        num_particles = size * size
        print(f"\n--- Grid {size}×{size} = {num_particles} particles ---")
        
        verts, faces = generate_grid_mesh(size)
        temp_obj = f"_temp_grid_{size}.obj"
        export_obj(np.array(verts), faces, temp_obj)
        
        result = {'size': size, 'particles': num_particles}
        
        try:
            cloth_cpu = MeshCloth(temp_obj, scale=1.0, offset=(0, 5, 0))
            result['springs'] = len(cloth_cpu.spring_indices)
            cpu_ms, _ = benchmark_cpu_no_selfcoll(cloth_cpu, Config.dt, warmup=3, iterations=20)
            result['cpu_ms'] = cpu_ms
            print(f"  CPU:  {cpu_ms:.4f} ms/step")
        except Exception as e:
            print(f"  CPU failed: {e}")
            result['cpu_ms'] = None
        
        if CUDA_AVAILABLE:
            try:
                cloth_cuda = MeshClothCUDA(temp_obj, scale=1.0, offset=(0, 5, 0))
                cloth_cuda.enable_self_collision = False
                cuda_ms, _ = benchmark_cuda(cloth_cuda, Config.dt, warmup=3, iterations=20)
                result['cuda_ms'] = cuda_ms
                result['speedup'] = result['cpu_ms'] / cuda_ms if result['cpu_ms'] else None
                print(f"  CUDA: {cuda_ms:.4f} ms/step")
                if result['speedup']:
                    print(f"  Speedup: {result['speedup']:.2f}×")
            except Exception as e:
                print(f"  CUDA failed: {e}")
                result['cuda_ms'] = None
        
        results_exp1.append(result)
        if os.path.exists(temp_obj):
            os.remove(temp_obj)
    
    print("\n" + "-"*70)
    print("EXPERIMENT 1 SUMMARY: CUDA vs CPU (No Self-Collision)")
    print("-"*70)
    print(f"{'Size':>6} {'Particles':>10} {'Springs':>10} {'CPU(ms)':>12} {'CUDA(ms)':>12} {'Speedup':>10}")
    print("-"*70)
    for r in results_exp1:
        cpu_str = f"{r['cpu_ms']:.4f}" if r.get('cpu_ms') else "N/A"
        cuda_str = f"{r.get('cuda_ms', 0):.4f}" if r.get('cuda_ms') else "N/A"
        speedup_str = f"{r.get('speedup', 0):.2f}×" if r.get('speedup') else "N/A"
        print(f"{r['size']:>6} {r['particles']:>10} {r.get('springs', 0):>10} {cpu_str:>12} {cuda_str:>12} {speedup_str:>10}")
    
    # 实验2: 空间哈希 vs 暴力搜索 (纯CUDA)
    if CUDA_AVAILABLE:
        print("\n\n" + "="*70)
        print("EXPERIMENT 2: Spatial Hash vs Brute-Force (CUDA Only)")
        print("="*70)
        print("对比自碰撞检测算法：O(n²) vs O(n×k)")
        
        results_exp2 = []
        
        for size in sizes_exp2:
            num_particles = size * size
            print(f"\n--- Grid {size}×{size} = {num_particles} particles ---")
            
            verts, faces = generate_grid_mesh(size)
            temp_obj = f"_temp_collision_{size}.obj"
            export_obj(np.array(verts), faces, temp_obj)
            
            try:
                cloth_cuda = MeshClothCUDA(temp_obj, scale=1.0, offset=(0, 5, 0))              
                print(f"  collision_thickness: {cloth_cuda.collision_thickness:.4f}")
                if hasattr(cloth_cuda, 'cell_size'):
                    print(f"  cell_size: {cloth_cuda.cell_size:.4f}")
                
                collision_result = benchmark_collision_methods_only(cloth_cuda, iterations=50)
                collision_result['size'] = size
                collision_result['particles'] = num_particles
                results_exp2.append(collision_result)
                
            except Exception as e:
                print(f"  Failed: {e}")
            finally:
                if os.path.exists(temp_obj):
                    os.remove(temp_obj)
        
        print("\n" + "-"*70)
        print("EXPERIMENT 2 SUMMARY: Spatial Hash vs Brute-Force")
        print("-"*70)
        print(f"{'Size':>6} {'Particles':>10} {'Brute(ms)':>12} {'Hash(ms)':>12} {'Speedup':>10}")
        print("-"*70)
        for r in results_exp2:
            brute_str = f"{r.get('brute_ms', 0):.4f}" if r.get('brute_ms') else "N/A"
            hash_str = f"{r.get('hash_ms', 0):.4f}" if r.get('hash_ms') else "N/A"
            speedup_str = f"{r.get('speedup', 0):.2f}×" if r.get('speedup') else "N/A"
            print(f"{r['size']:>6} {r['particles']:>10} {brute_str:>12} {hash_str:>12} {speedup_str:>10}")
    
    return results_exp1, results_exp2 if CUDA_AVAILABLE else None


def generate_grid_mesh(size, edge_length=0.1):
    """
    生成平面网格
    edge_length: 固定的边长，保持粒子密度恒定
    """
    verts = []
    faces = []
    for z in range(size):
        for x in range(size):
            verts.append([x * edge_length, 0, z * edge_length])
    for z in range(size - 1):
        for x in range(size - 1):
            p1 = z * size + x
            p2 = p1 + 1
            p3 = (z + 1) * size + x
            p4 = p3 + 1
            faces.append([p1, p3, p2])
            faces.append([p2, p3, p4])
    return verts, faces


def benchmark_collision_methods_only(cloth_cuda, iterations=30):
    """仅测试自碰撞方法性能（空间哈希 vs 暴力搜索）"""
    num_particles = cloth_cuda.particles.num
    threads_per_block = 256
    blocks = (num_particles + threads_per_block - 1) // threads_per_block
    
    result = {}
    
    # 预热暴力搜索kernel
    for _ in range(3):
        self_collision_kernel[blocks, threads_per_block](
            cloth_cuda.particles.pos, cloth_cuda.particles.vel, cloth_cuda.particles.inv_mass,
            cloth_cuda.adjacency_gpu, cloth_cuda.collision_thickness, num_particles
        )
    cuda.synchronize()
    
    # 预热空间哈希
    if hasattr(cloth_cuda, 'use_spatial_hash') and cloth_cuda.use_spatial_hash:
        for _ in range(3):
            cloth_cuda._update_spatial_hash()
            self_collision_spatial_hash_kernel[blocks, threads_per_block](
                cloth_cuda.particles.pos, cloth_cuda.particles.vel, cloth_cuda.particles.inv_mass,
                cloth_cuda.adjacency_gpu, cloth_cuda.cell_indices, cloth_cuda.sorted_indices,
                cloth_cuda.cell_offsets, cloth_cuda.cell_counts,
                cloth_cuda.collision_thickness, cloth_cuda.cell_size, cloth_cuda.grid_size,
                num_particles, cloth_cuda.num_cells
            )
        cuda.synchronize()
    
    # 1. 暴力搜索
    brute_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        self_collision_kernel[blocks, threads_per_block](
            cloth_cuda.particles.pos, cloth_cuda.particles.vel, cloth_cuda.particles.inv_mass,
            cloth_cuda.adjacency_gpu, cloth_cuda.collision_thickness, num_particles
        )
        cuda.synchronize()
        brute_times.append(time.perf_counter() - t0)
    
    brute_avg = np.median(brute_times) * 1000
    result['brute_ms'] = brute_avg
    print(f"  Brute-Force: {brute_avg:.4f} ms  ({num_particles}² = {num_particles**2} comparisons)")
    
    # 2. 空间哈希
    if hasattr(cloth_cuda, 'use_spatial_hash') and cloth_cuda.use_spatial_hash:
        hash_times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            cloth_cuda._update_spatial_hash()
            self_collision_spatial_hash_kernel[blocks, threads_per_block](
                cloth_cuda.particles.pos, cloth_cuda.particles.vel, cloth_cuda.particles.inv_mass,
                cloth_cuda.adjacency_gpu, cloth_cuda.cell_indices, cloth_cuda.sorted_indices,
                cloth_cuda.cell_offsets, cloth_cuda.cell_counts,
                cloth_cuda.collision_thickness, cloth_cuda.cell_size, cloth_cuda.grid_size,
                num_particles, cloth_cuda.num_cells
            )
            cuda.synchronize()
            hash_times.append(time.perf_counter() - t0)
        
        hash_avg = np.median(hash_times) * 1000
        result['hash_ms'] = hash_avg
        result['speedup'] = brute_avg / hash_avg
        print(f"  Spatial Hash: {hash_avg:.4f} ms  (grid {cloth_cuda.grid_size}³)")
        print(f"  Speedup: {result['speedup']:.2f}×")
    else:
        print(f"  Spatial Hash: N/A (not enabled for this mesh size)")
        result['hash_ms'] = None
        result['speedup'] = None
    
    return result


def plot_results(results_exp1, results_exp2):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Cloth Simulation Performance Benchmark', fontsize=14, fontweight='bold')
    
    if results_exp1:
        particles = [r['particles'] for r in results_exp1]
        cpu_times = [r['cpu_ms'] for r in results_exp1]
        cuda_times = [r.get('cuda_ms', 0) for r in results_exp1]
        speedups = [r.get('speedup', 0) for r in results_exp1]
        
        # 图1: 时间对比 (左上)
        ax1 = axes[0, 0]
        x = np.arange(len(particles))
        width = 0.35
        bars1 = ax1.bar(x - width/2, cpu_times, width, label='CPU (NumPy)', color='#3498db', alpha=0.8)
        bars2 = ax1.bar(x + width/2, cuda_times, width, label='CUDA', color='#e74c3c', alpha=0.8)
        
        ax1.set_xlabel('Particles', fontsize=11)
        ax1.set_ylabel('Time (ms/step)', fontsize=11)
        ax1.set_title('Experiment 1: CUDA vs CPU Execution Time', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'{p:,}' for p in particles])
        ax1.legend(loc='upper left')
        ax1.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars1, cpu_times):
            ax1.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=8)
        for bar, val in zip(bars2, cuda_times):
            ax1.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=8)
        
        # 图2: 加速比 (右上)
        ax2 = axes[0, 1]
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(speedups)))
        bars = ax2.bar(x, speedups, color=colors, edgecolor='black', linewidth=0.5)
        ax2.axhline(y=1, color='gray', linestyle='--', linewidth=1, label='Baseline (1×)')
        
        ax2.set_xlabel('Particles', fontsize=11)
        ax2.set_ylabel('Speedup (×)', fontsize=11)
        ax2.set_title('Experiment 1: CUDA Speedup vs CPU', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'{p:,}' for p in particles])
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, speedups):
            ax2.annotate(f'{val:.1f}×', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    if results_exp2:
        valid_results = [r for r in results_exp2 if r.get('hash_ms') is not None]
        
        if valid_results:
            particles2 = [r['particles'] for r in valid_results]
            brute_times = [r['brute_ms'] for r in valid_results]
            hash_times = [r['hash_ms'] for r in valid_results]
            speedups2 = [r['speedup'] for r in valid_results]
            
            # 图3: 时间对比 (左下)
            ax3 = axes[1, 0]
            x2 = np.arange(len(particles2))
            
            ax3.plot(x2, brute_times, 'o-', color='#e74c3c', linewidth=2, markersize=8, 
                    label=r'Brute-Force $O(n^2)$')
            ax3.plot(x2, hash_times, 's-', color='#2ecc71', linewidth=2, markersize=8, 
                    label=r'Spatial Hash $O(n \cdot k)$')
            
            # 标注理论复杂度曲线 - 用最后一个点拟合，使曲线更贴合实测数据
            if len(particles2) >= 2:
                p_ref = particles2[-1]
                t_ref = brute_times[-1]
                theoretical_n2 = [t_ref * (p/p_ref)**2 for p in particles2]
                ax3.plot(x2, theoretical_n2, '--s', color='#9b59b6', alpha=0.8, markersize=5,
                        markerfacecolor='none', label=r'$O(n^2)$ theoretical')
            
            ax3.set_xlabel('Particles', fontsize=11)
            ax3.set_ylabel('Time (ms)', fontsize=11)
            ax3.set_title('Experiment 2: Spatial Hash vs Brute-Force', fontsize=12)
            ax3.set_xticks(x2)
            ax3.set_xticklabels([f'{p:,}' for p in particles2])
            ax3.legend(loc='upper left')
            ax3.grid(alpha=0.3)
            
            # 图4: Hash加速比 (右下)
            ax4 = axes[1, 1]
            colors2 = ['#e74c3c' if s < 1 else '#2ecc71' for s in speedups2]
            bars2 = ax4.bar(x2, speedups2, color=colors2, edgecolor='black', linewidth=0.5)
            ax4.axhline(y=1, color='gray', linestyle='--', linewidth=1.5, label='Break-even (1×)')
            
            ax4.set_xlabel('Particles', fontsize=11)
            ax4.set_ylabel('Speedup (×)', fontsize=11)
            ax4.set_title('Experiment 2: Spatial Hash Speedup', fontsize=12)
            ax4.set_xticks(x2)
            ax4.set_xticklabels([f'{p:,}' for p in particles2])
            ax4.legend()
            ax4.grid(axis='y', alpha=0.3)
            
            for bar, val in zip(bars2, speedups2):
                color = '#e74c3c' if val < 1 else '#2ecc71'
                ax4.annotate(f'{val:.2f}×', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                            ha='center', va='bottom', fontsize=10, fontweight='bold', color=color)
        else:
            axes[1, 0].text(0.5, 0.5, 'No valid spatial hash data', ha='center', va='center', 
                           transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 1].text(0.5, 0.5, 'No valid spatial hash data', ha='center', va='center', 
                           transform=axes[1, 1].transAxes, fontsize=12)
    
    plt.tight_layout()
    
    output_path = 'benchmark_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n[Chart saved to {output_path}]")
    
    plt.show()


def save_raw_data(results_exp1, results_exp2):
    import csv
    
    csv_path1 = 'benchmark_exp1_cuda_vs_cpu.csv'
    with open(csv_path1, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Size', 'Particles', 'Springs', 'CPU_ms', 'CUDA_ms', 'Speedup'])
        for r in results_exp1:
            writer.writerow([
                r['size'], r['particles'], r.get('springs', 0),
                f"{r.get('cpu_ms', 0):.4f}" if r.get('cpu_ms') else 'N/A',
                f"{r.get('cuda_ms', 0):.4f}" if r.get('cuda_ms') else 'N/A',
                f"{r.get('speedup', 0):.2f}" if r.get('speedup') else 'N/A'
            ])
    print(f"[Raw data saved to {csv_path1}]")
    
    if results_exp2:
        csv_path2 = 'benchmark_exp2_hash_vs_brute.csv'
        with open(csv_path2, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Size', 'Particles', 'Brute_ms', 'Hash_ms', 'Speedup'])
            for r in results_exp2:
                writer.writerow([
                    r['size'], r['particles'],
                    f"{r.get('brute_ms', 0):.4f}" if r.get('brute_ms') else 'N/A',
                    f"{r.get('hash_ms', 0):.4f}" if r.get('hash_ms') else 'N/A',
                    f"{r.get('speedup', 0):.2f}" if r.get('speedup') else 'N/A'
                ])
        print(f"[Raw data saved to {csv_path2}]")


if __name__ == "__main__":
    import sys
    
    print("\n[Running two independent experiments]")
    print("[Experiment 1: CUDA vs CPU acceleration]")
    print("[Experiment 2: Spatial Hash vs Brute-Force acceleration]\n")
    
    results = test_scaling()

    if results:
        results_exp1, results_exp2 = results
        save_raw_data(results_exp1, results_exp2)
        plot_results(results_exp1, results_exp2)
