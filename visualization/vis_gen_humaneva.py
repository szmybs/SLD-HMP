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
from models.motion_pred_ours import *
from models import LinNF
from utils import util

from visualization.vis_pose import plt_row, plt_row_independent_save, plt_row_mixtures
from visualization.vis_skeleton import VisSkeleton


def denomarlize(*data):
    out = []
    for x in data:
        x = x * dataset.std + dataset.mean
        out.append(x)
    return out


def get_prediction(data, algo, sample_num, num_seeds=1, concat_hist=True, z=None):
    dct_m, idct_m = util.get_dct_matrix(t_pred + t_his)
    dct_m_all = dct_m.float().to(device)
    idct_m_all = idct_m.float().to(device)
    parts = cfg.nf_specs['parts']
    n_parts = len(parts)
    idx_pad = list(range(t_his)) + [t_his - 1] * t_pred
    traj_np = data[..., 1:, :].transpose([0, 2, 3, 1])  # .reshape(traj_np.shape[0], traj_np.shape[1], -1)
    traj = tensor(traj_np, device=device, dtype=dtype)  # .permute(0, 2, 1).contiguous()
    bs, nj, _, _ = traj.shape
    inp = traj.reshape([bs, -1, traj.shape[-1]]).transpose(1, 2)
    inp = torch.matmul(dct_m_all[:cfg.n_pre], inp[:, idx_pad, :]).transpose(1, 2). \
        reshape([bs, nj, 3, -1]).reshape([bs, nj, -1])
    inp = inp.unsqueeze(1).repeat([1, cfg.nk, 1, 1]).reshape([bs * cfg.nk, nj, -1])

    if algo == 'gcn':
        z = torch.randn([sample_num * num_seeds, n_parts, cfg.nf_specs['nz']], dtype=dtype, device=device)
        if args.fixlower:
            z[:, 0] = z[:1, 0]
        # z[:, :1] = z[:1, :1]
        Y = models['gcn'](inp, z)
        Y = Y.reshape([Y.shape[0], Y.shape[1], 3, cfg.n_pre]).reshape(
            [Y.shape[0], Y.shape[1] * 3, cfg.n_pre]).transpose(1, 2)
        Y = torch.matmul(idct_m_all[:, :cfg.n_pre], Y[:, :cfg.n_pre]).transpose(1, 2)[:, :, t_his:]
        X = traj[..., :t_his].reshape([traj.shape[0], traj.shape[1] * 3, t_his]).repeat([sample_num * num_seeds, 1, 1])

    if concat_hist:
        Y = torch.cat((X, Y), dim=-1)
    Y = Y.permute(0, 2, 1).contiguous().cpu().numpy()

    if Y.shape[0] > 1:
        Y = Y.reshape(-1, sample_num, Y.shape[-2], Y.shape[-1])
    else:
        Y = Y[None, ...]
    return Y


def get_gt(data):
    gt = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)
    return gt[:, t_his:, :]


def visualize():
    total_num = 0

    vis_skeleton = VisSkeleton(parents=[-1, 0, 1, 2, 3, 1, 5, 6, 0, 8, 9, 0, 11, 12, 1],
                                joints_left=[2, 3, 4, 8, 9, 10],
                                joints_right=[5, 6, 7, 11, 12, 13])

    if args.action != 'all':
        save_subdir = list(args.action)[0]
    else:
        save_subdir = args.action
    save_dir = os.path.join(os.getcwd(), 'output/imgs/HumanEva', save_subdir)
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
            pred = get_prediction(data, algo, sample_num=cfg.nk, num_seeds=num_seeds, concat_hist=False)
            
            pz = np.zeros(shape=(pred.shape[0], pred.shape[1], pred.shape[2], 45))
            pz[..., 3:] = pred
            pz = np.reshape(pz, newshape=(pz.shape[0], pz.shape[1], pz.shape[2], 15, 3))
            
            gz = np.zeros(shape=(gt.shape[0], gt.shape[1], 45))
            gz[..., 3:] = gt
            gz = np.reshape(gz, newshape=(gz.shape[0], gz.shape[1], 15, 3))

            '''
            for j in range(pz.shape[0]):
                pos_mixtures = []
                for k in range(10):
                    pos_mixtures.append(pz[j, k, -1, :, :])
                
                plt_row_independent_save(
                # plt_row(
                    skeleton = vis_skeleton,
                    pose = [gz[j, -1, :, :]],
                    mixtures = pos_mixtures,
                    type = "3D",
                    lcolor = "#3498db", rcolor = "#e74c3c",
                    view = (0, 0, 0),
                    # view = (0, -180, -90),
                    titles = None,
                    add_labels = False, 
                    only_pose = True,
                    save_dir = save_dir, 
                    save_name = 'GSPS'+'_'+str(total_num)
                )
                total_num += 1
            '''
            
            y = pz[:, :, [11, 23, 35, 47, 59]]
            y = np.swapaxes(y, 1, 2)
            x_pred = gz[:, [11, 23, 35, 47, 59]]

            for j in range(y.shape[0]):
                mixtures_lists = []
                for p in range(y.shape[1]):
                    mixtures_lists.append([])
                    for q in range(y.shape[2]):
                        mixtures_lists[p].append(y[j, p, q])
                
                plt_row_mixtures(
                    skeleton = vis_skeleton,
                    pose = mixtures_lists,
                    type = "3D",
                    lcolor = "#3498db", rcolor = "#e74c3c",
                    view = (0, 0, 0),
                    titles = None,
                    add_labels = False, 
                    only_pose = True,
                    save_dir = save_dir, 
                    save_name = 'GSPS_' + str(total_num) + '_mix'
                )

                poses = [x_pred[j,k] for k in range(x_pred.shape[1])]
                plt_row_mixtures(
                    skeleton = vis_skeleton,
                    pose = poses,
                    type = "3D",
                    lcolor = "#3498db", rcolor = "#e74c3c",
                    view = (0, 0, 0),
                    titles = None,
                    add_labels = False, 
                    only_pose = True,
                    save_dir = save_dir, 
                    save_name = 'GSPS_' + str(total_num)
                )
                total_num += 1




if __name__ == '__main__':
    all_algos = ['gcn']
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',
                        default='humaneva')
    parser.add_argument('--mode', default='vis')
    parser.add_argument('--data', default='test')
    parser.add_argument('--action', default='all')
    parser.add_argument('--num_seeds', type=int, default=1)
    parser.add_argument('--multimodal_threshold', type=float, default=0.5)
    parser.add_argument('--multimodal_th_high', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--n_pre', type=int, default=10)
    parser.add_argument('--n_his', type=int, default=5)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--nk', type=int, default=-1)
    parser.add_argument('--fixlower', action='store_true', default=False)
    parser.add_argument('--num_coupling_layer', type=int, default=4)
    for algo in all_algos:
        parser.add_argument('--iter_%s' % algo, type=int, default=500)
    args = parser.parse_args()

    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) if args.gpu_index >= 0 and \
                                                           torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    torch.set_grad_enabled(False)
    cfg = Config(args.cfg)
    logger = create_logger(os.path.join(cfg.log_dir, 'log_eval.txt'))

    algos = []
    for algo in all_algos:
        algos.append(algo)
    vis_algos = algos.copy()

    if args.action != 'all':
        args.action = set(args.action.split(','))

    """parameter"""
    if args.mode == 'vis':
        cfg.nk = 10
    else:
        if args.nk > 0:
            cfg.nk = args.nk
        else:
            cfg.nk = 50
    nz = cfg.nz
    nk = cfg.nk
    t_his = cfg.t_his
    t_pred = cfg.t_pred
    n_his = args.n_his
    cfg.n_his = n_his
    # n_pre = args.n_pre
    if 'n_pre' not in cfg.nf_specs.keys():
        n_pre = args.n_pre
    else:
        n_pre = cfg.nf_specs['n_pre']
    cfg.n_pre = n_pre
    cfg.num_coupling_layer = args.num_coupling_layer

    """data"""
    dataset_cls = DatasetH36M if cfg.dataset == 'h36m' else DatasetHumanEva
    dataset = dataset_cls(args.data, t_his, t_pred, actions=args.action, use_vel=cfg.use_vel,
                          multimodal_path=cfg.nf_specs[
                              'multimodal_path'] if 'multimodal_path' in cfg.nf_specs.keys() else None,
                          data_candi_path=cfg.nf_specs[
                              'data_candi_path'] if 'data_candi_path' in cfg.nf_specs.keys() else None)

    """models"""
    model_generator = {
        'gcn': get_model
    }
    models = {}
    for algo in algos:
        models[algo], pose_prior = model_generator[algo](cfg, dataset.traj_dim // 3, args.cfg)
        models[algo].float()
        model_path = getattr(cfg, f"vae_model_path") % getattr(args, f'iter_{algo}')
        print(f'loading {algo} model from checkpoint: {model_path}')
        model_cp = pickle.load(open(model_path, "rb"))
        models[algo].load_state_dict(model_cp['model_dict'])
        models[algo].to(device)
        models[algo].eval()

        LinNF.LinNF(data_dim=dataset.traj_dim, num_layer=3)
        cp_path = './results/h36m_nf/models/vae_0025.p' if cfg.dataset == 'h36m' else \
            './results/humaneva_nf/models/vae_0025.p'
        print('loading model from checkpoint: %s' % cp_path)
        model_cp = pickle.load(open(cp_path, "rb"))
        pose_prior.load_state_dict(model_cp['model_dict'])
        pose_prior.to(device)
        pose_prior.eval()

    if cfg.normalize_data:
        dataset.normalize_data(model_cp['meta']['mean'], model_cp['meta']['std'])

    visualize()