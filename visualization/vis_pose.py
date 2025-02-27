import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import argparse
import torch

import os, sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())

from visualization.vis_preprocess import coordinates_alignment


def show2Dpose(skeleton, pose, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=False, only_pose=False):
    parents = skeleton._parents
    connections = parents[1:]

    JL = skeleton._joints_left
    JR = skeleton._joints_right
    
    pose = np.squeeze(pose)
    assert pose.shape[1] == 2 or pose.shape[1] == 3 
    if pose.shape[1] == 3:
      pose = pose[:, 1:]
    
    y_min = np.min(pose[:, 1])
    pose -= np.array([0, y_min])
      
    # Make connection matrix
    for op, ed in enumerate(connections, 1):
      x, y = [np.array([pose[op, j], pose[ed, j]]) for j in range(2)]
      ax.plot(x, y, lw=2, c=lcolor if op in JL else rcolor)#lw线条宽度

    RADIUS = 1.0 # space around the subject
    # RADIUS = np.max(np.abs(pose))
    xroot, yroot = pose[0,0], pose[0,1]  #hip的位置
    ax.set_xlim([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim([0, 2 * RADIUS])
    ax.grid(True)

    if add_labels:
      ax.set_xlabel("x")
      ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5 * RADIUS))
      ax.set_ylabel("y")
      ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5 * RADIUS))  
    else:
      # Get rid of the ticks and tick labels 
      ax.set_xticks([])
      ax.set_yticks([])
      ax.get_xaxis().set_ticklabels([])
      ax.get_yaxis().set_ticklabels([])

    if only_pose:
      #Turn the x- and y-axis off.This affects the axis lines, ticks, ticklabels, grid and axis labels.
      ax.set_axis_off() 
    return



def show3Dpose(skeleton, pose, ax, max_radius, lcolor="#3498db", rcolor="#e74c3c", alpha=1.0, lw=2.0,
              add_labels=False, only_pose=False, **kwargs): # blue, orange
    parents = skeleton._parents
    connections = parents[1:]
    
    JL, JR = skeleton._joints_left, skeleton._joints_right
    # xy_radius, z_radius = max_radius
    xy_radius, z_radius = 0.85, 0.85
    
    '''
    pose = np.squeeze(pose)
    _pose = np.copy(pose)
    pose[1:, :] += pose[:1, :]
    
    z_min = np.min(pose[:, 2])
    pose -= np.array([0, 0, z_min])
    ''' 
    pose = np.squeeze(pose)
    tmp_pose = np.copy(pose)
    tmp_pose[1:, :] += tmp_pose[:1, :]
    
    z_min = np.min(tmp_pose[:, 2])
    tmp_pose -= np.array([0, 0, z_min])    
    
    '''
    inline_colors = {
      0:'black',    1:'lightcoral',   2:'red',    3:'chocolate',    4:'gold',
      5:'yellow',   6:'lawngreen',    7:'cyan',   8:'navy',   9:'blueviolet',
      10:'purple',    11:'deeppink',    12:'silver',    13:'green',
      14:'deepskyblue',   15:'white'
    }
    '''  
    
    # Make connection matrix
    for op, ed in enumerate(connections, 1):
      x, y, z = [np.array([tmp_pose[op, j], tmp_pose[ed, j]]) for j in range(3)]
      ax.plot(x, y, z, lw=lw, c=lcolor if op in JL else rcolor, alpha=alpha)#lw线条宽度
      
    # xroot, yroot, zroot = _pose[0,0], _pose[0,1], _pose[0,2] #hip的位置
    xroot, yroot, zroot = pose[0,0], pose[0,1], pose[0,2] #hip的位置
    ax.set_xlim3d([-xy_radius+xroot, xy_radius+xroot])
    ax.set_ylim3d([-xy_radius+yroot, xy_radius+yroot])
    ax.set_zlim3d([0, 2*z_radius])
    ax.grid(True)
    ax.set_box_aspect(((xy_radius, xy_radius, z_radius)))

    if add_labels:
      ax.set_xlabel("x")
      ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5 * xy_radius))
      
      ax.set_ylabel("y")
      ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5 * xy_radius))
       
      ax.set_zlabel("z")
      ax.zaxis.set_major_locator(ticker.MultipleLocator(0.5 * z_radius))
    else:
      # Get rid of the ticks and tick labels 
      ax.set_xticks([])
      ax.set_yticks([])
      ax.set_zticks([])
      
      ax.get_xaxis().set_ticklabels([])
      ax.get_yaxis().set_ticklabels([])
      ax.get_yaxis().set_ticklabels([])

    if only_pose:
      #Turn the x- and y-axis off.This affects the axis lines, ticks, ticklabels, grid and axis labels.
      ax.set_axis_off() 
      
    # Get rid of the panes (actually, make them white)
    white = (1.0, 1.0, 1.0, 0.0)
    # Get rid of the lines in 3d
    ax.xaxis.line.set_color(white)
    ax.yaxis.line.set_color(white)
    ax.zaxis.line.set_color(white)
    return



''' # 可以被plt_row代替
def plt_one(skeleton, pose, type="3D", lcolor="#3498db", rcolor="#e74c3c", titles=None, add_labels=False, only_pose=False,
            save_dir=None, save_name=None):
    fig = plt.figure()
    
    if type == "3D":
      ax = fig.add_subplot(projection='3d')
      show3Dpose(skeleton, pose, ax, lcolor=lcolor, rcolor=rcolor, add_labels=add_labels, only_pose=only_pose)
    elif type == "2D":
      ax = fig.add_subplot()
      show2Dpose(skeleton, pose, ax, lcolor=lcolor, rcolor=rcolor, add_labels=add_labels, only_pose=only_pose)
      
    if titles is not None:
      ax.set_title(titles, fontsize=16)    
    
    ax.view_init(45, 45)
    
    if save_dir is None or save_name is None:
      plt.show()
    else:
      plt.savefig(os.path.join(save_dir, save_name))
      plt.close()
    return
'''



def plt_row(skeleton, pose, mixtures=None, type="3D", lcolor="#3498db", rcolor="#e74c3c", view=(0, 90), alpha=1.0, lw=2.0,
            titles=None, add_labels=False, only_pose=False, save_dir=None, save_name=None, **kwargs):
    if isinstance(pose, list) is False:
        pose = list(pose) 
    cols = len(pose)
    _cols = cols+1 if mixtures else cols

    if "dpi" in kwargs:
        fig = plt.figure(figsize=[3*_cols, 6], dpi=kwargs["dpi"])
    else:
        fig = plt.figure(figsize=[3*_cols, 6], dpi=200)
    
    if mixtures:
        xy_radius_max, z_radius_max = coordinates_alignment(pose+mixtures)
    else:
        xy_radius_max, z_radius_max = coordinates_alignment(pose)
        
    for i in range(1, cols+1):
        if type ==  '3D':
          ax = fig.add_subplot(1, _cols, i, projection='3d')
          show3Dpose(skeleton, pose[i-1], ax, (xy_radius_max, z_radius_max), lcolor=lcolor, rcolor=rcolor, alpha=alpha, lw=lw, \
                     add_labels=add_labels, only_pose=only_pose)
        elif type == '2D':
          ax = fig.add_subplot(1, _cols, i)
          show2Dpose(skeleton, pose[i-1], ax, lcolor=lcolor, rcolor=rcolor, add_labels=add_labels, only_pose=only_pose) 
       
        if titles is not None:
            ax.set_title(titles[i-1], fontsize=10)
        ax.view_init(view[0], view[1]) if len(view) == 2 else ax.view_init(view[0], view[1], view[2])
    
    if mixtures:
        if type ==  '3D':
          ax = fig.add_subplot(1, _cols, _cols, projection='3d')
          for m in mixtures:
            show3Dpose(skeleton, m, ax, (xy_radius_max, z_radius_max), lcolor=lcolor, rcolor=rcolor, alpha=alpha, lw=lw, \
                      add_labels=add_labels, only_pose=only_pose)
        elif type == '2D':
          ax = fig.add_subplot(1, _cols, _cols)
          for m in mixtures:
            show2Dpose(skeleton, m, ax, lcolor=lcolor, rcolor=rcolor, add_labels=add_labels, only_pose=only_pose) 
            
        if titles is not None:
            ax.set_title(titles[i-1], fontsize=10)
        ax.view_init(view[0], view[1])
    
    # plt.tight_layout()
    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0)
    
    if save_dir is None or save_name is None:
        plt.show()
    else:
        plt.savefig(os.path.join(save_dir, save_name), pad_inches=0.0, bbox_inches='tight')
        plt.close()
    return



def plt_row_independent_save(skeleton, pose, mixtures=None, type="3D", lcolor="#3498db", rcolor="#e74c3c", view=(0, 90, 0), alpha=1.0, lw=2.0,
                             titles=None, add_labels=False, only_pose=False, save_dir=None, save_name=None, **kwargs):
    if isinstance(pose, list) is False:
        pose = list(pose) 
    cols = len(pose)
    _cols = cols+1 if mixtures else cols
    
    if mixtures:
        xy_radius_max, z_radius_max = coordinates_alignment(pose+mixtures)
    else:
        xy_radius_max, z_radius_max = coordinates_alignment(pose)
        
    for i in range(1, cols+1):
        if type ==  '3D':
          fig = plt.figure(figsize=[3, 3], dpi=300)
          ax = fig.add_subplot(1, 1, 1, projection='3d')
          show3Dpose(skeleton, pose[i-1], ax, (xy_radius_max, z_radius_max), lcolor=lcolor, rcolor=rcolor, alpha=alpha, lw=lw, \
                    add_labels=add_labels, only_pose=only_pose)
          plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0)
                 
        if titles is not None:
            ax.set_title(titles[i-1], fontsize=10)
        ax.view_init(view[0], view[1], view[2])
        
        name = save_name+"_"+str(i)+".jpg"
        plt.savefig(os.path.join(save_dir, name), pad_inches=0.0, bbox_inches='tight')
        plt.close()
    
    if mixtures:
        if type ==  '3D':
          fig = plt.figure(figsize=[3, 3], dpi=300)
          ax = fig.add_subplot(1, 1, 1, projection='3d')
          for m in mixtures:
            show3Dpose(skeleton, m, ax, (xy_radius_max, z_radius_max), lcolor=lcolor, rcolor=rcolor, alpha=alpha, lw=lw, \
                      add_labels=add_labels, only_pose=only_pose)
          plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0)
            
        if titles is not None:
            ax.set_title(titles[i-1], fontsize=10)
        ax.view_init(view[0], view[1], view[2])
        
        name = save_name+"_mix.jpg"
        plt.savefig(os.path.join(save_dir, name), pad_inches=0.0, bbox_inches='tight')
        plt.close()
    return      
  
  
def plt_row_mixtures(skeleton, pose, type="3D", lcolor="#3498db", rcolor="#e74c3c", 
                     view=(0, 90), alpha=1.0, lw=2.0,
                     titles=None, add_labels=False, only_pose=False, save_dir=None, save_name=None, **kwargs):
    if not isinstance(pose, list):
        pose = list(pose) 
    cols = len(pose)
    
    if "dpi" in kwargs:
        fig = plt.figure(figsize=[3*cols, 6], dpi=kwargs["dpi"])
    else:
        fig = plt.figure(figsize=[3*cols, 6], dpi=200)
      
    if "vis_points" in kwargs:
        vis_points = kwargs["vis_points"]
    else:
        vis_points = None
    
    if "alphas" in kwargs:
        alphas = kwargs["alphas"]
    else:
        alphas = 1.0
    
    pose_np = np.array(pose)
    pose_np = np.reshape(pose_np, newshape=(-1, pose_np.shape[-2], pose_np.shape[-1]))
    xy_radius_max, z_radius_max = coordinates_alignment(pose_np.tolist())
    
    for i in range(1, cols+1):
        if type == '3D':
          ax = fig.add_subplot(1, cols, i, projection='3d')
          
          if isinstance(pose[i-1], list):
            for j, m in enumerate(pose[i-1]):
              m_alpha = alphas[i-1][j] if isinstance(alphas, list) else alphas
              show3Dpose(skeleton, m, ax, (xy_radius_max, z_radius_max), lcolor=lcolor, rcolor=rcolor, alpha=m_alpha, lw=lw, \
                        add_labels=add_labels, only_pose=only_pose, vis_points=vis_points, fig=fig)            
            
          else:
            show3Dpose(skeleton, pose[i-1], ax, (xy_radius_max, z_radius_max), lcolor=lcolor, rcolor=rcolor, alpha=alpha, lw=lw, \
                       add_labels=add_labels, only_pose=only_pose)
        else:
          pass            
                    
        if titles is not None:
            ax.set_title(titles[i-1], fontsize=10)
        ax.view_init(view[0], view[1])
    
    # plt.tight_layout()
    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0)
    
    if save_dir is None or save_name is None:
        plt.show()
    else:
        plt.savefig(os.path.join(save_dir, save_name), pad_inches=0.0, bbox_inches='tight')
        plt.close()
    return  