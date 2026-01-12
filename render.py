import bpy
import os
import math
import mathutils

OBJ_DIR = "D:\\1\\acg\\collision_sim\\output"
SPHERE_FILE = os.path.join(OBJ_DIR, "sphere_pos.txt")
COLLIDER_FILE = os.path.join(OBJ_DIR, "collider_transform.txt")
SOLID_MESH_PATH = "D:\\1\\acg\\collision_sim\\bunny_200_subdivided_1.obj"
SOLID_MESH_SCALE = 20.0
SOLID_INITIAL_ROTATION = (0.0, 0.0, 0.0)

GROUND_SIZE = 1000.0 
CHECKER_SQUARE_SIZE = 2.0 

CAMERA_DISTANCE = 20 
CAMERA_HEIGHT = 20
CAMERA_LENS = 50.0

ROTATION_CORRECTION_DEG = ( 90.0, 0.0, 0.0 )
POSITION_CORRECTION = (0.0, 0.0, 0.0)
FLIP_Z = True

def sim_euler_to_blender_euler(rx_deg, ry_deg, rz_deg):
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)

    Rx = mathutils.Matrix(((1.0, 0.0, 0.0),
                           (0.0, math.cos(rx), -math.sin(rx)),
                           (0.0, math.sin(rx),  math.cos(rx))))
    Ry = mathutils.Matrix((( math.cos(ry), 0.0, math.sin(ry)),
                           ( 0.0,        1.0, 0.0        ),
                           (-math.sin(ry), 0.0, math.cos(ry))))
    Rz = mathutils.Matrix((( math.cos(rz), -math.sin(rz), 0.0),
                           ( math.sin(rz),  math.cos(rz), 0.0),
                           ( 0.0,           0.0,          1.0)))
    R_sim = Rz @ Ry @ Rx

    M = mathutils.Matrix(((1.0, 0.0, 0.0),
                          (0.0, 0.0,-1.0),
                          (0.0, 1.0, 0.0)))
    R_blender = M @ R_sim @ M.transposed()

    cx, cy, cz = (math.radians(ROTATION_CORRECTION_DEG[0]),
                  math.radians(ROTATION_CORRECTION_DEG[1]),
                  math.radians(ROTATION_CORRECTION_DEG[2]))
    R_corr = mathutils.Euler((cx, cy, cz), 'XYZ').to_matrix().to_3x3()
    R_blender = R_corr @ R_blender

    eul = R_blender.to_euler('XYZ')
    return eul.x, eul.y, eul.z

def clear_scene():
    print("Clearing old meshes...")
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.select_set(True)
    bpy.ops.object.delete()

def setup_camera():
    print("Setting up Camera...")
    
    if "Camera" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Camera"], do_unlink=True)
    
    camera_data = bpy.data.cameras.new(name="Camera")
    camera_obj = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera_obj)
    bpy.context.scene.camera = camera_obj
    
    camera_obj.location = (CAMERA_DISTANCE, -CAMERA_DISTANCE * 0.5, CAMERA_HEIGHT)
    
    direction = mathutils.Vector((0, 0, 0)) - mathutils.Vector(camera_obj.location)
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera_obj.rotation_euler = rot_quat.to_euler()
    
    camera_data.lens = CAMERA_LENS
    
    print(f"Camera set at {camera_obj.location} with focal length {CAMERA_LENS}mm")

def setup_lighting():
    print("Setting up Lighting...")
    
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type == 'LIGHT':
            obj.select_set(True)
    bpy.ops.object.delete()

    bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
    sun = bpy.context.active_object
    sun.name = "Sun_Key_Light"
    
    sun.data.energy = 2.0 
    sun.data.angle = 0.1 
    
    import math
    sun.rotation_euler = (math.radians(45), 0, -math.radians(45))

    bpy.ops.object.light_add(type='AREA', location=(0, 0, 15))
    top_light = bpy.context.active_object
    top_light.name = "Top_Fill_Light"
    top_light.data.energy = 60.0
    top_light.data.size = 10.0

    bpy.ops.object.light_add(type='AREA', location=(0, -6, 3))
    rim = bpy.context.active_object
    rim.name = "Rim_Light"
    rim.data.energy = 25.0
    rim.data.size = 3.0
    rim.rotation_euler = (math.radians(60), 0, math.radians(180))

def setup_world_environment():
    world = bpy.context.scene.world
    try:
        if not world.use_nodes:
            world.use_nodes = True
    except:
        pass

    nodes = world.node_tree.nodes
    links = world.node_tree.links

    bg_node = nodes.get("Background")
    if not bg_node:
        bg_node = nodes.new(type="ShaderNodeBackground")
        output = nodes.get("World Output") or nodes.new(type="ShaderNodeOutputWorld")
        links.new(bg_node.outputs['Background'], output.inputs['Surface'])

    bg_node.inputs['Color'].default_value = (0.1, 0.1, 0.1, 1.0)
    bg_node.inputs['Strength'].default_value = 1.2

def get_or_create_material(name, setup_func):
    mat = bpy.data.materials.get(name)
    if not mat:
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        setup_func(mat)
    return mat

def setup_silk_material(mat):
    nodes = mat.node_tree.nodes
    principled = nodes.get("Principled BSDF")
    if principled:
        principled.inputs['Base Color'].default_value = (0.6, 0.1, 0.1, 1) 
        principled.inputs['Roughness'].default_value = 0.6 
        if 'Specular' in principled.inputs:
            principled.inputs['Specular'].default_value = 0.2
        for key in ['Sheen Weight', 'Sheen']:
            if key in principled.inputs:
                principled.inputs[key].default_value = 1.0
                break
    mat.diffuse_color = (0.8, 0.05, 0.1, 1)

def setup_metal_material(mat):
    nodes = mat.node_tree.nodes
    principled = nodes.get("Principled BSDF")
    if principled:
        principled.inputs['Base Color'].default_value = (0.9, 0.9, 0.9, 1)
        principled.inputs['Metallic'].default_value = 1.0 
        principled.inputs['Roughness'].default_value = 0.1
    mat.diffuse_color = (0.8, 0.8, 0.8, 1)

def setup_solid_material(mat):
    nodes = mat.node_tree.nodes
    principled = nodes.get("Principled BSDF")
    if principled:
        principled.inputs['Base Color'].default_value = (0.45, 0.45, 0.5, 1)
        principled.inputs['Metallic'].default_value = 0.1 
        principled.inputs['Roughness'].default_value = 0.75
        if 'Specular' in principled.inputs:
            principled.inputs['Specular'].default_value = 0.2
    mat.diffuse_color = (0.6, 0.6, 0.65, 1)

def setup_checker_material(mat):
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    principled = nodes.get("Principled BSDF")
    
    checker = nodes.new(type="ShaderNodeTexChecker")
    checker.inputs['Scale'].default_value = GROUND_SIZE / CHECKER_SQUARE_SIZE
    checker.inputs['Color1'].default_value = (0.15, 0.15, 0.15, 1) 
    checker.inputs['Color2'].default_value = (0.4, 0.4, 0.4, 1) 
    
    links.new(checker.outputs['Color'], principled.inputs['Base Color'])
    principled.inputs['Roughness'].default_value = 0.6

def import_solid_mesh(scale=1.0):
    if not os.path.exists(SOLID_MESH_PATH):
        print(f"Warning: Solid mesh {SOLID_MESH_PATH} not found, skipping.")
        return
    
    print(f"Importing solid mesh: {SOLID_MESH_PATH} (scale={scale})")
    
    bpy.ops.wm.obj_import(filepath=SOLID_MESH_PATH)
    
    imported_objs = bpy.context.selected_objects
    if not imported_objs:
        print("Failed to import solid mesh!")
        return
    
    solid_obj = imported_objs[0]
    solid_obj.name = "Solid_Collider"
    
    solid_obj.scale = (scale, scale, scale)
    
    bpy.ops.object.shade_smooth()
    
    mat = get_or_create_material("Solid_Gray_Metal", setup_solid_material)
    if solid_obj.data.materials:
        solid_obj.data.materials[0] = mat
    else:
        solid_obj.data.materials.append(mat)
    
    sub = solid_obj.modifiers.new(name="Subdivision", type='SUBSURF')
    sub.render_levels = 2
    sub.levels = 1
    
    if os.path.exists(COLLIDER_FILE):
        print(f"Loading collider animation from {COLLIDER_FILE}...")
        with open(COLLIDER_FILE, 'r') as f:
            lines = f.readlines()
        
        frame_count = 0
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            try:
                data = list(map(float, line.split()))
            except ValueError:
                continue
            
            if len(data) < 6: continue
            
            x, y, z, rx, ry, rz = data[0], data[1], data[2], data[3], data[4], data[5]
            
            bz = -z if FLIP_Z else z
            solid_obj.location = (x + POSITION_CORRECTION[0],
                                   bz + POSITION_CORRECTION[1],
                                   y + POSITION_CORRECTION[2])

            total_rx = SOLID_INITIAL_ROTATION[0] + rx
            total_ry = SOLID_INITIAL_ROTATION[1] + ry
            total_rz = SOLID_INITIAL_ROTATION[2] + rz
            
            solid_obj.rotation_mode = 'XYZ'
            ex, ey, ez = sim_euler_to_blender_euler(total_rx, total_ry, total_rz)
            solid_obj.rotation_euler = (ex, ey, ez)
            
            solid_obj.keyframe_insert(data_path="location", frame=frame_count)
            solid_obj.keyframe_insert(data_path="rotation_euler", frame=frame_count)
            frame_count += 1
        
        print(f"Collider animation applied ({frame_count} frames)")
    else:
        print(f"Note: {COLLIDER_FILE} not found - solid mesh is static.")
    
    print(f"Solid mesh imported: {solid_obj.name}")



def create_checkerboard_ground():
    bpy.ops.mesh.primitive_plane_add(size=GROUND_SIZE, location=(0, -0.02, 0))
    ground = bpy.context.active_object
    ground.name = "Ground_Plane"
    
    mat = get_or_create_material("Checker_Floor", setup_checker_material)
    if ground.data.materials:
        ground.data.materials[0] = mat
    else:
        ground.data.materials.append(mat)

def create_animated_sphere():
    if not os.path.exists(SPHERE_FILE):
        print(f"Note: {SPHERE_FILE} not found - no animated sphere will be rendered.")
        print("      This is normal when using static mesh collider.")
        return

    print(f"Loading sphere animation from {SPHERE_FILE}...")
    bpy.ops.mesh.primitive_uv_sphere_add(radius=2, location=(0,0,0))
    sphere_obj = bpy.context.active_object
    sphere_obj.name = "Simulated_Sphere"
    bpy.ops.object.shade_smooth()
    
    mat = get_or_create_material("Metal_Shiny", setup_metal_material)
    if sphere_obj.data.materials:
        sphere_obj.data.materials[0] = mat
    else:
        sphere_obj.data.materials.append(mat)

    with open(SPHERE_FILE, 'r') as f:
        lines = f.readlines()

    for frame_idx, line in enumerate(lines):
        data = list(map(float, line.strip().split()))
        if len(data) < 3: continue
        
        x, y, z = data[0], data[1], data[2]
        
        sphere_obj.location = (x, -z, y)
        
        sphere_obj.keyframe_insert(data_path="location", frame=frame_idx)

def import_cloth_sequence():
    files = sorted([f for f in os.listdir(OBJ_DIR) if f.endswith(".obj")])
    
    if not files:
        print("No .obj files found!")
        return

    collection_name = "ClothSimulation"
    if collection_name in bpy.data.collections:
        collection = bpy.data.collections[collection_name]
    else:
        collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(collection)
    
    silk_mat = get_or_create_material("Silk_Fabric", setup_silk_material)

    for frame_idx, filename in enumerate(files):
        filepath = os.path.join(OBJ_DIR, filename)
        
        bpy.ops.wm.obj_import(filepath=filepath)
        
        imported_objs = bpy.context.selected_objects
        if not imported_objs: continue
        
        obj = imported_objs[0]
        obj.name = f"Cloth_Frame_{frame_idx}"
        
        for other_col in obj.users_collection:
            other_col.objects.unlink(obj)
        collection.objects.link(obj)
        
        if obj.data.materials:
            obj.data.materials[0] = silk_mat
        else:
            obj.data.materials.append(silk_mat)
            

        sub = obj.modifiers.new(name="Subdivision", type='SUBSURF')
        sub.render_levels = 2
        sub.levels = 1

        solid = obj.modifiers.new(name="Solidify", type='SOLIDIFY')
        solid.thickness = 0.04
        solid.offset = 0
        
        for poly in obj.data.polygons:
            poly.use_smooth = True
            
        obj.hide_viewport = True
        obj.hide_render = True
        obj.keyframe_insert(data_path="hide_viewport", frame=frame_idx-1)
        obj.keyframe_insert(data_path="hide_render", frame=frame_idx-1)
        
        obj.hide_viewport = False
        obj.hide_render = False
        obj.keyframe_insert(data_path="hide_viewport", frame=frame_idx)
        obj.keyframe_insert(data_path="hide_render", frame=frame_idx)
        
        obj.hide_viewport = True
        obj.hide_render = True
        obj.keyframe_insert(data_path="hide_viewport", frame=frame_idx+1)
        obj.keyframe_insert(data_path="hide_render", frame=frame_idx+1)

    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = len(files) - 1

if __name__ == "__main__":
    bpy.context.scene.render.engine = 'CYCLES'
    try:
        bpy.context.scene.cycles.device = 'GPU'
    except:
        pass
    clear_scene()
    setup_camera()
    setup_lighting()
    setup_world_environment()
    try:
        bpy.context.scene.view_settings.view_transform = 'Filmic'
        bpy.context.scene.view_settings.look = 'Medium High Contrast'
        bpy.context.scene.view_settings.exposure = -0.4
        bpy.context.scene.view_settings.gamma = 1.0
    except Exception:
        pass
    create_checkerboard_ground()
    create_animated_sphere()
    #import_solid_mesh(scale=SOLID_MESH_SCALE)
    import_cloth_sequence()

    
#& "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" --python render.py