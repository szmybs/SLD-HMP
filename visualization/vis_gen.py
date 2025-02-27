import sys
import os
import argparse
import pickle

if __name__ == "__main__":
    sys.path.append(os.getcwd())
    sys.path.append('./Motion')

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')


from utils import *
from motion_pred.utils.config import Config
from motion_pred.utils.dataset_h36m import DatasetH36M
from motion_pred.utils.dataset_humaneva import DatasetHumanEva
from models.motion_pred import *

from visualization.vis_pose import plt_row, plt_row_independent_save
from visualization.vis_skeleton import VisSkeleton


def denomarlize(*data):
    out = []
    for x in data:
        x = x * dataset.std + dataset.mean
        out.append(x)
    return out


def get_prediction(data, algo, sample_num, num_seeds=1, concat_hist=True):
    traj_np = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)
    traj = tensor(traj_np, device=device, dtype=dtype).permute(1, 0, 2).contiguous()
    X = traj[:t_his]

    if algo == 'dlow':
        X = X.repeat((1, num_seeds, 1))
        Z_g = models[algo].sample(X)
        X = X.repeat_interleave(nk, dim=1)
        Y = models['vae'].decode(X, Z_g)
    elif algo == 'vae':
        X = X.repeat((1, sample_num * num_seeds, 1))
        Y = models[algo].sample_prior(X)

    if concat_hist:
        Y = torch.cat((X, Y), dim=0)
    Y = Y.permute(1, 0, 2).contiguous().cpu().numpy()
    if Y.shape[0] > 1:
        Y = Y.reshape(-1, sample_num, Y.shape[-2], Y.shape[-1])
    else:
        Y = Y[None, ...]
    return Y


def get_gt(data):
    gt = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)
    return gt[:, t_his:, :]


def visualize():
    traj_idx = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    total_num = 0

    vis_skeleton = VisSkeleton(parents=[-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12,
                                        16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
                                joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
                                joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])  
    removed_joints = {4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31}
    vis_skeleton.remove_joints(removed_joints)
    vis_skeleton.adjust_connection_manually(([11, 8], [14, 8]))

    if args.action != 'all':
        save_subdir = list(args.action)[0]
    else:
        save_subdir = args.action
    save_dir = os.path.join(os.getcwd(), 'output/imgs/', save_subdir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("save_dir:" + str(save_dir))
    
    data_gen = dataset.iter_generator(step=cfg.t_his)
    num_samples = 0
    num_seeds = args.num_seeds
    for i, data in enumerate(data_gen):
        num_samples += 1
        gt = get_gt(data)
        for algo in algos:
            if algo != 'dlow':
                continue
            
            pred = get_prediction(data, algo, sample_num=cfg.nk, num_seeds=num_seeds, concat_hist=False)
            # pred = pred[:, traj_idx]
            
            pz = np.zeros(shape=(pred.shape[0], pred.shape[1], pred.shape[2], 51))
            pz[..., 3:] = pred
            pz = np.reshape(pz, newshape=(pz.shape[0], pz.shape[1], pz.shape[2], 17, 3))
            
            gz = np.zeros(shape=(gt.shape[0], gt.shape[1], 51))
            gz[..., 3:] = gt
            gz = np.reshape(gz, newshape=(gz.shape[0], gz.shape[1], 17, 3))

            for j in range(pz.shape[0]):
                pos_mixtures = []
                for k in range(len(traj_idx)):
                    pos_mixtures.append(pz[j, k, -1, :, :])
                
                plt_row_independent_save(
                # plt_row(
                    skeleton = vis_skeleton,
                    pose = [gz[j, -1, :, :]],
                    mixtures = pos_mixtures,
                    type = "3D",
                    lcolor = "#3498db", rcolor = "#e74c3c",
                    # view = (90, -180, -90),
                    view = (0, -180, -90),
                    titles = None,
                    add_labels = False, 
                    only_pose = True,
                    save_dir = save_dir, 
                    save_name = str(total_num)
                )
                total_num += 1




if __name__ == '__main__':
    all_algos = ['dlow', 'vae']
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='h36m_nsamp10')
    parser.add_argument('--mode', default='stats')
    parser.add_argument('--data', default='test')
    parser.add_argument('--action', default='all')
    parser.add_argument('--num_seeds', type=int, default=1)
    parser.add_argument('--multimodal_threshold', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_index', type=int, default=-1)
    for algo in all_algos:
        parser.add_argument('--iter_%s' % algo, type=int, default=None)
    args = parser.parse_args()

    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) if args.gpu_index >= 0 and torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    torch.set_grad_enabled(False)
    cfg = Config(args.cfg)
    logger = create_logger(os.path.join(cfg.log_dir, 'log_eval.txt'))

    algos = []
    for algo in all_algos:
        iter_algo = 'iter_%s' % algo
        num_algo = 'num_%s_epoch' % algo
        setattr(args, iter_algo, getattr(cfg, num_algo))
        algos.append(algo)
    vis_algos = algos.copy()

    if args.action != 'all':
        args.action = set(args.action.split(','))

    """parameter"""
    nz = cfg.nz
    nk = cfg.nk
    t_his = cfg.t_his
    t_pred = cfg.t_pred

    """data"""
    dataset_cls = DatasetH36M if cfg.dataset == 'h36m' else DatasetHumanEva
    dataset = dataset_cls(args.data, t_his, t_pred, actions=args.action, use_vel=cfg.use_vel)

    """models"""
    model_generator = {
        'vae': get_vae_model,
        'dlow': get_dlow_model,
    }
    models = {}
    for algo in algos:
        models[algo] = model_generator[algo](cfg, dataset.traj_dim)
        model_path = getattr(cfg, f"{algo}_model_path") % getattr(args, f'iter_{algo}')
        print(f'loading {algo} model from checkpoint: {model_path}')
        model_cp = pickle.load(open(model_path, "rb"))
        models[algo].load_state_dict(model_cp['model_dict'])
        models[algo].to(device)
        models[algo].eval()

    if cfg.normalize_data:
        dataset.normalize_data(model_cp['meta']['mean'], model_cp['meta']['std'])

    visualize()