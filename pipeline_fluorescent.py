from PIL import Image, ImageSequence
from shutil import rmtree
import os
import imageio
import cv2
import numpy as np
from sklearn.preprocessing import normalize

for sample_n in range(2, 3):
    dir = "temp"
    points = []
    frame = []
    n_layers = 0
    red_factor = 4
    voxel_size = [1.73, 1.73, 3]
    scale = 0.005

    if os.path.isdir(dir): rmtree(dir)
    os.mkdir(dir)

    im = Image.open('data/clump%d.tif'%sample_n)
    for i, page in enumerate(ImageSequence.Iterator(im)):
        page.save("temp/page%03d.png"%i)

    with imageio.get_writer("cotton.gif", mode="I") as writer:
        images = os.listdir(dir)
        for layer, file in enumerate(images):
            frame = cv2.imread(os.path.join(dir, file), cv2.IMREAD_GRAYSCALE)
            norm_img = np.zeros(frame.size)
            #norm_img = cv2.normalize(frame, norm_img, 0, 255, cv2.NORM_MINMAX)
            #mask = cv2.inRange(norm_img, 50, 255)
            #norm_img = cv2.bitwise_and(norm_img, norm_img, mask=mask)
            small_img = cv2.resize(frame, (0,0), fx=1/red_factor, fy=1/red_factor) 
            print(layer)
            #writer.append_data(norm_img)

            pointx, pointy = np.where(small_img != 0) 
            for i in range(len(pointx)):
                points.append([pointx[i]*voxel_size[0]*red_factor, pointy[i]*voxel_size[1]*red_factor, voxel_size[2]*(len(images) - layer), small_img[pointx[i]][pointy[i]]])

            n_layers += 1

    points = np.array(points)
    color = points[:, 3]
    normalized_color = (color-np.min(color))/(np.max(color)-np.min(color))
    points[:, 3] = normalized_color*255
    points = points[points[:, 3] > 70]

    pcd = "# .PCD v.7 - Point Cloud Data file format\n" 
    pcd += "VERSION .7\n" 
    pcd += "FIELDS x y z rgb\n" 
    pcd += "SIZE 4 4 4 4\n" 
    pcd += "TYPE F F F F\n" 
    pcd += "COUNT 1 1 1 1\n" 
    pcd += "WIDTH %d\n" %len(frame)
    pcd += "HEIGHT %d\n" %(n_layers*5)
    pcd += "VIEWPOINT 0 0 0 1 0 0 0\n" 
    pcd += "POINTS %d\n" %len(points) 
    pcd += "DATA ascii\n" 

    ply = "ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n" %len(points)

    for point in points:
        pcd += "%f %f %f %d\n" %(point[0]*scale, point[2]*scale, point[1]*scale, int(point[3]))
        ply += "%f %f %f %d %d %d\n" %(point[0]*scale, point[2]*scale, point[1]*scale, int(point[3]), int(point[3]), int(point[3]))

    file = open("clump%d.pcd"%sample_n, "w")
    file.write(pcd)
    file.close()

    file = open("clump%d.ply"%sample_n, "w")
    file.write(ply)
    file.close()