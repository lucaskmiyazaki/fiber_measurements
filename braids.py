import math
import numpy as np

def generate_ply_file(points, file_name):
    header = f"ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
    point_cloud = ""

    for point in points:
        x, y, z = point
        point_cloud += f"{x} {y} {z}\n"

    with open(file_name, 'w') as ply_file:
        ply_file.write(header + point_cloud)

    print(f"PLY file '{file_name}' generated successfully.")

def generate_center_line_points(num_points, twist_factor, diameter):
    rope1_points = []
    rope2_points = []

    for i in range(num_points):
        t = i * twist_factor

        x1 = i
        y1 = 1.05 * diameter * math.cos(t)
        z1 = 1.05 * diameter * math.sin(t)

        x2 = i
        y2 = 1.05 * diameter * math.cos(t + math.pi)
        z2 = 1.05 * diameter * math.sin(t + math.pi)

        rope1_points.append((x1, y1, z1))
        rope2_points.append((x2, y2, z2))

    return rope1_points, rope2_points

def generate_surface_point_cloud(center_line_points, diameter):
    surface_points = []

    for point in center_line_points:
        x, y, z = point

        # Generate points in a circular pattern around the center line
        theta = np.linspace(0, 2*np.pi, 20)
        r = diameter / 2
        dx = r * np.cos(theta)
        dy = r * np.sin(theta)
        dz = np.zeros_like(theta)

        # Translate and add the circular points to the snake points
        surface_points.extend(zip(x + dx, y + dy, z + dz))

    return surface_points

num_points = 1000  # Number of points to generate
twist_factor = 0.02  # Adjust the twist factor to control the twisting rate
diameter = 10

rope1_points, rope2_points = generate_center_line_points(num_points, twist_factor, diameter)

generate_ply_file(rope1_points, "rope1.ply")
generate_ply_file(rope2_points, "rope2.ply")
