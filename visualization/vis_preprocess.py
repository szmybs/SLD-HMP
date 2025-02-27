import numpy as np
import os
import sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())
    

def coordinates_alignment(poses):
    xy_radius_list, z_radius_list = [], []   
    for pose in poses:
        _pose = np.squeeze(pose)
        _pose[1:, :] += _pose[:1, :]
        
        xroot, yroot, zroot = _pose[0, 0], _pose[0, 1], _pose[0, 2]
        xy_radius = max(np.max(np.abs(_pose[:,0]-xroot)), np.max(np.abs(_pose[:,1]-yroot)))
        z_radius = (np.max(_pose[:,2]) - np.min(_pose[:,2])) / 2
        
        xy_radius_list.append(xy_radius)
        z_radius_list.append(z_radius)
    xy_radius_max = np.max(np.array(xy_radius_list))
    z_radius_max = np.max(np.array(z_radius_list))
    return xy_radius_max, z_radius_max