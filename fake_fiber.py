import numpy as np
import math
import re

skeleton_density = 1
#pcd_density = 1
scale = 0.005

def generate_cylinder_pcd(point0, point1, point2, radius, wall_thickness, pcd_density):
    # Compute the vector between the two points
    current_line = np.array(point2) - np.array(point1)
    prev_line    = np.array(point1) - np.array(point0) if len(point0) > 0 else current_line
    
    # Compute the height and direction of the cylinder
    height = np.linalg.norm(current_line)
    direction = current_line / height
    base_normal = prev_line / np.linalg.norm(prev_line)
    num_points_h = int(abs(height * pcd_density))
    num_points_r = int(abs(math.pi * radius * pcd_density))
    num_points_t = int(wall_thickness * pcd_density)
    if (num_points_r < 1): num_points_r = 1 
    if (num_points_t < 1): num_points_t = 1 

    # Compute Perpendiculars
    base_transition = np.array(range(num_points_h))/num_points_h
    arbitrary_vector = np.cross(direction, base_normal)
    sin_angle = np.linalg.norm(arbitrary_vector) / (np.linalg.norm(direction) * np.linalg.norm(base_normal))
    if (sin_angle < 0.1):
        arbitrary_vector = np.array([1, 0, 0]) if np.abs(direction[0]) < 0.5 else np.array([0, 1, 0])
    normal_transition = base_normal[np.newaxis, :] * (1 - base_transition)[:, np.newaxis] + direction[np.newaxis, :] * base_transition[:, np.newaxis]
    arbitrary_vector = np.tile(arbitrary_vector, (num_points_h, 1))
    v1 = np.cross(normal_transition, arbitrary_vector)
    v2 = np.cross(v1, normal_transition)
    
    v1 /= np.tile(np.linalg.norm(v1, axis=1), (3, 1)).T
    v2 /= np.tile(np.linalg.norm(v2, axis=1), (3, 1)).T
    
    # Generate points on the circular base of the cylinder
    theta = np.linspace(0, 2*np.pi, num_points_r)
    r = np.linspace(radius - wall_thickness, radius, num_points_t)
    x_base = r[np.newaxis, np.newaxis, :] * (np.cos(theta)[np.newaxis, :, np.newaxis] * v1[:, 0][:, np.newaxis, np.newaxis] + np.sin(theta)[np.newaxis, :, np.newaxis] * v2[:, 0][:, np.newaxis, np.newaxis])
    y_base = r[np.newaxis, np.newaxis, :] * (np.cos(theta)[np.newaxis, :, np.newaxis] * v1[:, 1][:, np.newaxis, np.newaxis] + np.sin(theta)[np.newaxis, :, np.newaxis] * v2[:, 1][:, np.newaxis, np.newaxis])
    z_base = r[np.newaxis, np.newaxis, :] * (np.cos(theta)[np.newaxis, :, np.newaxis] * v1[:, 2][:, np.newaxis, np.newaxis] + np.sin(theta)[np.newaxis, :, np.newaxis] * v2[:, 2][:, np.newaxis, np.newaxis])
    
    # Generate points on the cylinder surface by translating and rotating the base points
    displacement = np.linspace(0, height, num_points_h)
    x = point1[0] + direction[0] * displacement[:, np.newaxis, np.newaxis] + x_base #[np.newaxis, :] * (1-base_transition[:, np.newaxis]) + x_base1[np.newaxis, :] * base_transition[:, np.newaxis]
    y = point1[1] + direction[1] * displacement[:, np.newaxis, np.newaxis] + y_base #[np.newaxis, :] * (1-base_transition[:, np.newaxis]) + y_base1[np.newaxis, :] * base_transition[:, np.newaxis]
    z = point1[2] + direction[2] * displacement[:, np.newaxis, np.newaxis] + z_base #[np.newaxis, :] * (1-base_transition[:, np.newaxis]) + z_base1[np.newaxis, :] * base_transition[:, np.newaxis]
    
    # Flatten the coordinate arrays and combine them into a point cloud
    point_cloud = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
    
    return point_cloud

def get_rotation_matrix(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    v = np.cross(v1, v2)
    s = np.linalg.norm(v)
    c = np.dot(v1, v2)
    skew_symmetric = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + skew_symmetric + np.dot(skew_symmetric, skew_symmetric) * ((1 - c) / (s ** 2))
    return rotation_matrix

def generate_half_sphere_pcd(n, p, radius, wall_thickness, pcd_density):
    num_points = int(abs(math.pi * radius * pcd_density))
    num_points_t = int(wall_thickness * pcd_density)

    n = -n / np.linalg.norm(n)

    # Generate points on the surface of a sphere
    u = np.linspace(0, np.pi / 2, int(num_points // 2))
    v = np.linspace(0, 2 * np.pi, num_points*2)
    r = np.linspace(radius - wall_thickness, radius, num_points_t)
    r, u, v = np.meshgrid(r, u, v)
    x = r * np.cos(u) * np.cos(v)
    y = r * np.cos(u) * np.sin(v)
    z = r * np.sin(u)

    # Rotate the points to align with the plane
    points = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
    rotation_matrix = get_rotation_matrix(np.array([0, 0, 1]), n)
    rotated_points = np.dot(points, rotation_matrix.T)

    translated_points = rotated_points + p
    return translated_points

def pcd_to_ply(points, file_name):
    header = f"ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
    point_cloud = ""

    for point in points:
        x, y, z = point
        point_cloud += f"{float(x)} {float(y)} {float(z)}\n"

    with open(file_name, 'w') as ply_file:
        ply_file.write(header + point_cloud)

    print(f"PLY file '{file_name}' generated successfully.")

def ply_to_skeleton(file_name):
    pattern = r'([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)'
    point_cloud = []

    with open(file_name, 'r') as file:
        for line in file:
            match = re.match(pattern, line)
            if match:
                x = float(match.group(1))
                y = float(match.group(2))
                z = float(match.group(3))
                point_cloud.append([x, y, z])

    return np.array(point_cloud)

def generate_pcd_from_skeleton(skeleton, diameter, pcd_density):
    radius = diameter / 2
    wall_thickness = np.random.rand() * radius / 2
    point0 = []
    pcd = []

    # generate tube
    for i in range(len(skeleton) - 1):
        point1 = skeleton[i] 
        point2 = skeleton[i+1] 
        cylinder = generate_cylinder_pcd(point0, point1, point2, radius, wall_thickness, pcd_density)
        pcd = np.concatenate([pcd, cylinder]) if len(pcd) > 0 else cylinder
        point0 = point1

    # generate tip
    tip1 = generate_half_sphere_pcd(skeleton[1] - skeleton[0], skeleton[0], radius, wall_thickness, pcd_density)
    tip2 = generate_half_sphere_pcd(skeleton[-2] - skeleton[-1], skeleton[-1], radius, wall_thickness, pcd_density)
    pcd = np.concatenate([pcd, tip1])
    pcd = np.concatenate([pcd, tip2])

    return pcd

def generate_braids_skeleton(points, twist_factor, diameter):
    rope1_points = []
    rope2_points = []

    ray = diameter / 2
    for i in range(len(points)):
        t = i * twist_factor 

        x1 = i + points[i][0]
        y1 = 1.05 * ray * math.cos(t) + points[i][1]
        z1 = 1.05 * ray * math.sin(t) + points[i][2]

        x2 = i + points[i][0]
        y2 = 1.05 * ray * math.cos(t + math.pi) + points[i][1]
        z2 = 1.05 * ray * math.sin(t + math.pi) + points[i][2]

        rope1_points.append([x1, y1, z1])
        rope2_points.append([x2, y2, z2])

    return np.array(rope1_points), np.array(rope2_points)

def generate_parabole_skeleton(length):
    rope1_points = []
    rope2_points = []
    rope3_points = []
    rope4_points = []
    rope5_points = []
    rope6_points = []

    density = 1
    num_points = length * density
    for i in range(num_points):
        feed = i / density

        x1 = feed
        y1 = length*((length - 2*feed) / length)**(2) 
        z1 = 0

        x2 = feed
        y2 = length*((length - 2*feed) / length)**(3)
        z2 = 0

        x3 = feed
        y3 = length*((feed) / length)**(4)
        z3 = 0

        x4 = feed
        y4 = length*abs((length - 2*feed) / length)
        z4 = 0

        x5 = feed
        y5 = length*math.sin(math.pi*(length - 2*feed) / length) *0.2
        z5 = 0

        x6 = feed
        y6_ant1 = rope6_points[i-1][1] if len(rope6_points) > 0 else 0.1*length / num_points
        y6_ant2 = rope6_points[i-2][1] if len(rope6_points) > 1 else 0
        y6 = (y6_ant1 + y6_ant2) 
        z6 = 0

        rope1_points.append([x1, y1, z1])
        rope2_points.append([x2, y2, z2])
        rope3_points.append([x3, y3, z3])
        rope4_points.append([x4, y4, z4])
        rope5_points.append([x5, y5, z5])
        rope6_points.append([x6, y6, z6])

    return [np.array(rope1_points), np.array(rope2_points), np.array(rope3_points), np.array(rope4_points), np.array(rope5_points), np.array(rope6_points)]

def generate_circle_thread(map, diameter, row, is_horizontal):
    knot_length = diameter *1.3
    num_points = int(knot_length * skeleton_density)
    if (map == 0).all(): return [], None
    rope_points = []
    link = None

    for col in range(-1, len(map)+1):
        if col == -1 or col == len(map): direction = 0
        else: 
            direction = map[col] if is_horizontal else -map[col]
            if (map[col:] == 0).all()  : 
                return np.array(rope_points), -(col)
            if (map[:col+1] == 0).all():
                rope_points = []
                link = col
                continue
        for i in range(num_points):
            feed = i / skeleton_density
            x = feed + (col+1) * knot_length if is_horizontal else     (row + 1.5) * knot_length 
            y = feed + (col+1) * knot_length if not is_horizontal else (row + 1.5) * knot_length 
            z = direction * diameter * math.sin(i * math.pi / num_points)
            rope_points.append([x, y, z])

    return np.array(rope_points), link

def generate_parabole_thread(map, diameter, row, is_horizontal):
    knot_length = diameter *1.5
    num_points = int(knot_length * skeleton_density)
    if (map == 0).all(): return [], None
    rope_points = []
    link = None

    for col in range(-1, len(map)+1):
        if col == -1 or col == len(map): direction = 0
        else: 
            direction = map[col] if is_horizontal else -map[col]
            if (map[col:] == 0).all()  : 
                return np.array(rope_points), -(col)
            if (map[:col+1] == 0).all():
                rope_points = []
                link = col
                continue
        for i in range(num_points):
            if (direction == -1): direction = 0
            feed = i / skeleton_density
            x = feed + (col+1) * knot_length if is_horizontal else     (row + 1.5) * knot_length 
            y = feed + (col+1) * knot_length if not is_horizontal else (row + 1.5) * knot_length 
            z = 1.2*direction * diameter * (-((2*i - num_points)/num_points)**2 + 1)
            rope_points.append([x, y, z])

    return np.array(rope_points), link

def generate_circle_thread_skeleton(thread_map, diameter):
    ropes = []
    for row in range(len(thread_map)):
        rope_points, link = generate_circle_thread(thread_map[row], diameter, row, is_horizontal=True)
        if link is not None and link > 0: rope_points = np.flip(rope_points, axis=0)
        if len(rope_points) > 0: ropes.append(rope_points)

    transp_map = thread_map.transpose()
    for col in range(len(transp_map)):
        rope_points, link = generate_circle_thread(transp_map[col], diameter, col, is_horizontal=False)
        if link is not None: 
            if link < 0: 
                link = -link
                rope_points = np.flip(rope_points, axis=0)
            linked_rope = np.concatenate([ropes[link], rope_points])
            ropes[link] = linked_rope
        elif len(rope_points): ropes.append(rope_points)

    return ropes

def generate_parabole_thread_skeleton(thread_map, diameter):
    ropes = []
    for row in range(len(thread_map)):
        rope_points, link = generate_parabole_thread(thread_map[row], diameter, row, is_horizontal=True)
        if link is not None and link > 0: rope_points = np.flip(rope_points, axis=0)
        if len(rope_points) > 0: ropes.append(rope_points)

    transp_map = thread_map.transpose()
    for col in range(len(transp_map)):
        rope_points, link = generate_parabole_thread(transp_map[col], diameter, col, is_horizontal=False)
        if link is not None: 
            if link < 0: 
                link = -link
                rope_points = np.flip(rope_points, axis=0)
            linked_rope = np.concatenate([ropes[link], rope_points])
            ropes[link] = linked_rope
        elif len(rope_points): ropes.append(rope_points)

    return ropes

def disturb_skeleton(skeleton, diameter):
    threshold = diameter * 0.05

    deformed_skeleton = []
    num_points = len(skeleton)
    for i in range(num_points):
        point = skeleton[i]
        max_deformation = threshold
        deformation = np.random.uniform(-max_deformation, max_deformation, size=3)
        
        # Adjust deformation to respect the difference threshold between neighboring points
        if i > 0:
            prev_deformation = deformed_skeleton[i-1] - skeleton[i-1]
            deformation = (deformation + prev_deformation)/2

        deformed_point = point + deformation
        deformed_skeleton.append(deformed_point)
    
    return deformed_skeleton

def pcd_to_voxel_space(pcds, file_name):
    max_x = 0
    max_y = 0
    max_z = 0
    min_x = float('inf')
    min_y = float('inf')
    min_z = float('inf')

    for pcd in pcds:
        local_min_x = np.min(pcd[:, 0])
        local_min_y = np.min(pcd[:, 1])
        local_min_z = np.min(pcd[:, 2])
        local_max_x = np.max(pcd[:, 0])
        local_max_y = np.max(pcd[:, 1])
        local_max_z = np.max(pcd[:, 2])

        if local_min_x < min_x: min_x = local_min_x 
        if local_min_y < min_y: min_y = local_min_y 
        if local_min_z < min_z: min_z = local_min_z
        if local_max_x > max_x: max_x = local_max_x 
        if local_max_y > max_y: max_y = local_max_y 
        if local_max_z > max_z: max_z = local_max_z 

    voxel_space = np.zeros((math.ceil(max_x - min_x), math.ceil(max_y - min_y), math.ceil(max_z - min_z)))
    for pcd in pcds:
        for point in pcd:
            voxel_space[int(point[0] - min_x)][int(point[1] - min_y)][int(point[2] - min_z)] = 1

    with open(file_name, 'wb') as f:
        np.save(f, voxel_space)

def load_voxel_space(file_name):
    with open(file_name, 'rb') as f:
        a = np.load(f)
        return a

#skeleton = ply_to_skeleton("skeleton.ply")
#skeleton = skeleton / scale
#radius = 5 + 30*np.random.rand()
#fiber = generate_pcd_from_skeleton(skeleton, radius)
#pcd_to_voxel_space([fiber], "fiber.npy")
#a=load_voxel_space("fiber.npy")
#pcd_to_ply(fiber, "fiber.ply")


#skeleton1, skeleton2 = generate_parabole_skeleton(100)
#skeleton1, skeleton2 = generate_braids_skeleton(skeleton2, 0.2, 10)
#fiber1 = generate_pcd_from_skeleton(skeleton1, 10)
#fiber2 = generate_pcd_from_skeleton(skeleton2, 10)
#pcd_to_ply(fiber1, "fiber1.ply")
#pcd_to_ply(fiber2, "fiber2.ply")

#skeletons = generate_parabole_skeleton(100)
#for i in range(len(skeletons)):
#    deformed_skeleton = disturb_skeleton(skeletons[i], 10)
#    skeleton1, skeleton2 = generate_braids_skeleton(deformed_skeleton, 0.2, 10)
#    fiber1 = generate_pcd_from_skeleton(skeleton1, 10)
#    fiber2 = generate_pcd_from_skeleton(skeleton2, 10)
#    pcd_to_ply(fiber1, "fiber1-%d.ply"%i)
#    pcd_to_ply(fiber2, "fiber2-%d.ply"%i)
#    pcd_to_ply(skeleton1, "sk1-%d.ply"%i)
#    pcd_to_ply(skeleton2, "sk2-%d.ply"%i)

#thread_map = np.array([
#    [1, 0, -1, 1, -1],
#    [0, 0, 1, -1, -1],
#    [1, -1, 1, 1, 0],
#    [-1, 1, -1, 0, 0],
#    [-1, -1, 0, 0, 0],
#])
#skeletons = generate_circle_thread_skeleton(thread_map, 100, 10)
#for i in range(len(skeletons)):
#    deformed_skeleton = disturb_skeleton(skeletons[i], 10)
#    fiber = generate_pcd_from_skeleton(deformed_skeleton, 10)
#    pcd_to_ply(fiber, "fiber%d.ply"%i)
#    pcd_to_ply(deformed_skeleton, "sk%d.ply"%i)

#thread_map = np.array([
#    [1, 0, 1, -1],
#    [-1, 0, -1, -1],
#    [0, 0, 1, -1],
#    [-1, 1, -1, 1],
#])
#skeletons = generate_parabole_thread_skeleton(thread_map, 100, 10)
#for i in range(len(skeletons)):
#    deformed_skeleton = disturb_skeleton(skeletons[i], 10)
#    fiber = generate_pcd_from_skeleton(deformed_skeleton, 10)
#    pcd_to_ply(fiber, "fiber%d.ply"%i)
#    pcd_to_ply(deformed_skeleton, "sk%d.ply"%i)