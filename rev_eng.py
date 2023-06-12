import os
from fake_fiber import *

scale = 0.005
pcd_density = 1
dir_name = 'data' 
 
# Get the list of all files and directories
dir_list = os.listdir(dir_name)

fibers = []
for subdir in dir_list:
    file_list = os.listdir("%s/%s"%(dir_name, subdir))
    if not os.path.exists("training_data/%s"%subdir): os.mkdir("training_data/%s"%subdir)

    for index, file in enumerate(file_list):
        skeleton = ply_to_skeleton("data/%s/%s"%(subdir, file))
        skeleton = skeleton / scale
        pcd_to_ply(skeleton, "training_data/%s/skeleton%d.ply"%(subdir, index))
        radius = 5 + 30*np.random.rand()
        fibers.append(generate_pcd_from_skeleton(skeleton, radius, pcd_density))

    print(fibers)
    pcd_to_voxel_space(fibers, "training_data/%s/fiber.npy"%subdir)