import os
import sys
import pickle
import argparse
import time
from torch import maximum, optim
# from torch.utils.tensorboard import SummaryWriter
import itertools
sys.path.append(os.getcwd())
from utils import *
from motion_pred.utils.config import Config
from motion_pred.utils.dataset_h36m_multimodal import DatasetH36M
from motion_pred.utils.dataset_humaneva_multimodal import DatasetHumanEva
from motion_pred.utils.visualization import render_animation
from models.motion_pred import *
from utils import util, valid_angle_check
from utils.metrics import *
from FID.fid import fid
from FID.fid_classifier import classifier_fid_factory, classifier_fid_humaneva_factory
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import random
import re
import time
import math
from abc import ABC
from torch import Tensor


def kde(y, y_pred):
    y, y_pred = torch.from_numpy(y).float().to(torch.device('cuda')), torch.from_numpy(y_pred).float().to(torch.device('cuda'))
    bs, sp, ts, ns, d = y_pred.shape
    kde_ll = torch.zeros((bs, ts, ns), device=y_pred.device)

    for b in range(bs):
        for t in range(ts):
            for n in range(ns):
                try:
                    kernel = GaussianKDE(y_pred[b, :, t, n, :])
                except BaseException:
                    print("b: %d - t: %d - n: %d" % (b, t, n))
                    continue
                # pred_prob = kernel(y_pred[:, b, t, :, n])
                gt_prob = kernel(y[b, None, t, n, :])
                kde_ll[b, t, n] = gt_prob
    # mean_kde_ll = torch.mean(kde_ll)
    mean_kde_ll = torch.mean(torch.mean(kde_ll, dim=-1), dim=0)[None]
    return mean_kde_ll

  
class DynamicBufferModule(ABC, torch.nn.Module):
    """Torch module that allows loading variables from the state dict even in the case of shape mismatch."""
    
    def get_tensor_attribute(self, attribute_name: str) -> Tensor:
        """Get attribute of the tensor given the name.
        Args:
            attribute_name (str): Name of the tensor
        Raises:
            ValueError: `attribute_name` is not a torch Tensor
        Returns:
            Tensor: Tensor attribute
        """
        attribute = getattr(self, attribute_name)
        if isinstance(attribute, Tensor):
            return attribute
        raise ValueError(f"Attribute with name '{attribute_name}' is not a torch Tensor")

    def _load_from_state_dict(self, state_dict: dict, prefix: str, *args):
        """Resizes the local buffers to match those stored in the state dict.
        Overrides method from parent class.
        Args:
          state_dict (dict): State dictionary containing weights
          prefix (str): Prefix of the weight file.
          *args:
        """
        persistent_buffers = {k: v for k, v in self._buffers.items() if k not in self._non_persistent_buffers_set}
        local_buffers = {k: v for k, v in persistent_buffers.items() if v is not None}

        for param in local_buffers.keys():
            for key in state_dict.keys():
                if key.startswith(prefix) and key[len(prefix) :].split(".")[0] == param:
                    if not local_buffers[param].shape == state_dict[key].shape:
                        attribute = self.get_tensor_attribute(param)
                        attribute.resize_(state_dict[key].shape)
        super()._load_from_state_dict(state_dict, prefix, *args)
        

class GaussianKDE(DynamicBufferModule):
    """Gaussian Kernel Density Estimation.
    Args:
        dataset (Optional[Tensor], optional): Dataset on which to fit the KDE model. Defaults to None.
    """

    def __init__(self, dataset):
        super().__init__()

        self.register_buffer("bw_transform", Tensor())
        self.register_buffer("dataset", Tensor())
        self.register_buffer("norm", Tensor())
        
        if dataset is not None:
            self.fit(dataset)
        
        
    def forward(self, features: Tensor) -> Tensor:
        """Get the KDE estimates from the feature map.
        Args:
          features (Tensor): Feature map extracted from the CNN
        Returns: KDE Estimates
        """
        features = torch.matmul(features, self.bw_transform)

        estimate = torch.zeros(features.shape[0]).to(features.device)
        for i in range(features.shape[0]):
            embedding = ((self.dataset - features[i]) ** 2).sum(dim=1)
            embedding = self.log_norm - (embedding / 2)
            estimate[i] = torch.mean(embedding)
        return estimate


    def fit(self, dataset: Tensor) -> None:
        """Fit a KDE model to the input dataset.
        Args:
          dataset (Tensor): Input dataset.
        Returns:
            None
        """        
        num_samples, dimension = dataset.shape

        # compute scott's bandwidth factor
        factor = num_samples ** (-1 / (dimension + 4))

        cov_mat = self.cov(dataset.T)
        inv_cov_mat = torch.linalg.inv(cov_mat)
        inv_cov = inv_cov_mat / factor**2
        
        # transform data to account for bandwidth
        bw_transform = torch.linalg.cholesky(inv_cov)
        dataset = torch.matmul(dataset, bw_transform)
        
        #
        norm = torch.prod(torch.diag(bw_transform))
        norm *= math.pow((2 * math.pi), (-dimension / 2))

        self.bw_transform = bw_transform
        self.dataset = dataset
        self.norm = norm
        self.log_norm = torch.log(self.norm)
        return


    @staticmethod
    def cov(tensor: Tensor) -> Tensor:
        """Calculate the unbiased covariance matrix.
        Args:
            tensor (Tensor): Input tensor from which covariance matrix is computed.
        Returns:
            Output covariance matrix.
        """
        mean = torch.mean(tensor, dim=1, keepdim=True)
        cov = torch.matmul(tensor - mean, (tensor - mean).T) / (tensor.size(1) - 1)
        return cov



def recon_loss(Y_g, Y, Y_mm, Y_hg=None, Y_h=None):
    stat = torch.zeros(Y_g.shape[2])
    diff = Y_g - Y.unsqueeze(2) # TBMV
    dist = diff.pow(2).sum(dim=-1).sum(dim=0) # BM
    
    value, indices = dist.min(dim=1)

    loss_recon_1 = value.mean()
    
    diff = Y_hg - Y_h.unsqueeze(2) # TBMC
    loss_recon_2 = diff.pow(2).sum(dim=-1).sum(dim=0).mean()
    

    with torch.no_grad():
        ade = torch.norm(diff, dim=-1).mean(dim=0).min(dim=1)[0].mean()

    diff = Y_g[:, :, :, None, :] - Y_mm[:, :, None, :, :]
    
    mask = Y_mm.abs().sum(-1).sum(0) > 1e-6 

    dist = diff.pow(2) 
    with torch.no_grad():
        zeros = torch.zeros([dist.shape[1], dist.shape[2]], requires_grad=False).to(dist.device)# [b,m]    
        zeros.scatter_(dim=1, index=indices.unsqueeze(1).repeat(1, dist.shape[2]), src=zeros+dist.max()-dist.min()+1)
        zeros = zeros.unsqueeze(0).unsqueeze(3).unsqueeze(4)
        dist += zeros
    dist = dist.sum(dim=-1).sum(dim=0)

    value_2, indices_2 = dist.min(dim=1)
    loss_recon_multi = value_2[mask].mean()
    if torch.isnan(loss_recon_multi):
        loss_recon_multi = torch.zeros_like(loss_recon_1)
    
    mask = torch.tril(torch.ones([cfg.nk, cfg.nk], device=device)) == 0
    # TBMC
    
    yt = Y_g.reshape([-1, cfg.nk, Y_g.shape[3]]).contiguous()
    pdist = torch.cdist(yt, yt, p=1)[:, mask]
    return loss_recon_1, loss_recon_2, loss_recon_multi, ade, stat, (-pdist / 100).exp().mean()


def angle_loss(y):
    ang_names = list(valid_ang.keys())
    y = y.reshape([-1, y.shape[-1]])
    ang_cos = valid_angle_check.h36m_valid_angle_check_torch(
        y) if cfg.dataset == 'h36m' else valid_angle_check.humaneva_valid_angle_check_torch(y)
    loss = tensor(0, dtype=dtype, device=device)
    b = 1
    for an in ang_names:
        lower_bound = valid_ang[an][0]
        if lower_bound >= -0.98:
            # loss += torch.exp(-b * (ang_cos[an] - lower_bound)).mean()
            if torch.any(ang_cos[an] < lower_bound):
                # loss += b * torch.exp(-(ang_cos[an][ang_cos[an] < lower_bound] - lower_bound)).mean()
                loss += (ang_cos[an][ang_cos[an] < lower_bound] - lower_bound).pow(2).mean()
        upper_bound = valid_ang[an][1]
        if upper_bound <= 0.98:
            # loss += torch.exp(b * (ang_cos[an] - upper_bound)).mean()
            if torch.any(ang_cos[an] > upper_bound):
                # loss += b * torch.exp(ang_cos[an][ang_cos[an] > upper_bound] - upper_bound).mean()
                loss += (ang_cos[an][ang_cos[an] > upper_bound] - upper_bound).pow(2).mean()
    return loss


def loss_function(traj_est, traj, traj_multimodal, prior_lkh, prior_logdetjac, _lambda):
    lambdas = cfg.lambdas 
    nj = dataset.traj_dim // 3 

    Y_g = traj_est.view(traj_est.shape[0], traj.shape[1], traj_est.shape[1]//traj.shape[1], -1)[t_his:] # T B M V
    Y = traj[t_his:]
    Y_multimodal = traj_multimodal[t_his:]
    Y_hg=traj_est.view(traj_est.shape[0], traj.shape[1], traj_est.shape[1]//traj.shape[1], -1)[:t_his]
    Y_h= traj[:t_his]
    RECON, RECON_2, RECON_mm, ade, stat, JL = recon_loss(Y_g, Y, Y_multimodal,Y_hg, Y_h)
    # maintain limb length
    parent = dataset.skeleton.parents()
    tmp = traj[0].reshape([cfg.batch_size, nj, 3])
    pgt = torch.zeros([cfg.batch_size, nj + 1, 3], dtype=dtype, device=device)
    pgt[:, 1:] = tmp
    limbgt = torch.norm(pgt[:, 1:] - pgt[:, parent[1:]], dim=2)[None, :, None, :]
    tmp = traj_est.reshape([-1, cfg.batch_size, cfg.nk, nj, 3])
    pest = torch.zeros([tmp.shape[0], cfg.batch_size, cfg.nk, nj + 1, 3], dtype=dtype, device=device)
    pest[:, :, :, 1:] = tmp
    limbest = torch.norm(pest[:, :, :, 1:] - pest[:, :, :, parent[1:]], dim=4)
    loss_limb = torch.mean((limbgt - limbest).pow(2).sum(dim=3))

    # angle loss
    loss_ang = angle_loss(Y_g)
    if _lambda < 0.1:
        _lambda *= 10
    else:
        _lambda = 1
    
    loss_r =  loss_limb * lambdas[1] + JL * lambdas[3] * _lambda  + RECON * lambdas[4] + RECON_mm * lambdas[5] \
            - prior_lkh.mean() * lambdas[6] + RECON_2 * lambdas[7]# - prior_logdetjac.mean() * lambdas[7]
    if loss_ang > 0:
        loss_r += loss_ang * lambdas[8]
    return loss_r, np.array([loss_r.item(), loss_limb.item(), loss_ang.item(),
                            JL.item(), RECON.item(), RECON_2.item(), RECON_mm.item(), ade.item(),
                            prior_lkh.mean().item(), prior_logdetjac.mean().item()]), stat#, indices_key, indices_2_key
   

def dct_transform_torch(data, dct_m, dct_n):
    '''
    B, 60, 35
    '''
    batch_size, features, seq_len = data.shape

    data = data.contiguous().view(-1, seq_len)  # [180077*60， 35]
    data = data.permute(1, 0)  # [35, b*60]

    out_data = torch.matmul(dct_m[:dct_n, :], data)  # [dct_n, 180077*60]
    out_data = out_data.permute(1, 0).contiguous().view(-1, features, dct_n)  # [b, 60, dct_n]
    return out_data

def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m

def train(epoch, stats):

    dct_m, i_dct_m = get_dct_matrix(cfg.t_his+cfg.t_pred)
    dct_m = torch.from_numpy(dct_m).float().to(device)
    i_dct_m = torch.from_numpy(i_dct_m).float().to(device)

    model.train()
    t_s = time.time()
    train_losses = 0
    train_grad = 0
    total_num_sample = 0
    n_modality = 10
    loss_names = ['LOSS', 'loss_limb', 'loss_ang', 'loss_DIV',
                  'RECON', 'RECON_2', 'RECON_multi', "ADE", 'p(z)', 'logdet']
    generator = dataset.sampling_generator(num_samples=cfg.num_data_sample, batch_size=cfg.batch_size,
                                           n_modality=n_modality)
    prior = torch.distributions.Normal(torch.tensor(0, dtype=dtype, device=device),
                                       torch.tensor(1, dtype=dtype, device=device))
    
    for traj_np, traj_multimodal_np in tqdm(generator):
        with torch.no_grad():

            bs, _, nj, _ = traj_np[..., 1:, :].shape # [bs, t_full, numJoints, 3]
            traj_np = traj_np[..., 1:, :].reshape(traj_np.shape[0], traj_np.shape[1], -1) # bs, T, NumJoints*3
            traj = tensor(traj_np, device=device, dtype=dtype).permute(1, 0, 2).contiguous() # T, bs, NumJoints*3
            
            traj_multimodal_np = traj_multimodal_np[..., 1:, :]  # [bs, n_modality, t_full, NumJoints, 3]
            traj_multimodal_np = traj_multimodal_np.reshape([bs, n_modality, t_his + t_pred, -1]).transpose(
                [2, 0, 1, 3]) # [t_full, bs, n_modality, NumJoints*3]
            traj_multimodal = tensor(traj_multimodal_np, device=device, dtype=dtype)  # .permute(0, 2, 1).contiguous()
        
            X = traj[:t_his]
            Y = traj[t_his:]

        pred, a, b = model(traj)
        
        pred_tmp1 = pred.reshape([-1, pred.shape[-1] // 3, 3])
        pred_tmp = torch.zeros_like(pred_tmp1[:, :1, :])
        pred_tmp1 = torch.cat([pred_tmp, pred_tmp1], dim=1)
        pred_tmp1 = util.absolute2relative_torch(pred_tmp1, parents=dataset.skeleton.parents()).reshape(
            [-1, pred.shape[-1]])
        z, prior_logdetjac = pose_prior(pred_tmp1)
        prior_lkh = prior.log_prob(z).sum(dim=-1)
        
        loss, losses, stat = loss_function(pred.unsqueeze(2), traj, traj_multimodal, prior_lkh, prior_logdetjac, epoch / cfg.num_epoch)
        
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=100)
        train_grad += grad_norm
        optimizer.step()
        train_losses += losses
        
        total_num_sample += 1
        del loss
        
    scheduler.step()
    train_losses /= total_num_sample
    losses_str = ' '.join(['{}: {:.4f}'.format(x, y) for x, y in zip(loss_names, train_losses)])
    lr = optimizer.param_groups[0]['lr']
    # average cost of log time 20s
    # tb_logger.add_scalar('train_grad', train_grad / total_num_sample, epoch)
    # logger.info('====> Epoch: {} Time: {:.2f} {}  lr: {:.5f} branch_stats: {}'.format(epoch, time.time() - t_s, losses_str , lr, stats))
    
    return stats

def get_multimodal_gt(dataset_test):
    all_data = []
    data_gen = dataset_test.iter_generator(step=cfg.t_his)
    for data, _ in tqdm(data_gen):
        # print(data.shape)
        data = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)
        all_data.append(data)
    all_data = np.concatenate(all_data, axis=0)
    all_start_pose = all_data[:, t_his - 1, :]
    pd = squareform(pdist(all_start_pose))
    traj_gt_arr = []
    num_mult = []
    for i in range(pd.shape[0]):
        ind = np.nonzero(pd[i] < args.multimodal_threshold)
        traj_gt_arr.append(all_data[ind][:, t_his:, :])
        num_mult.append(len(ind[0]))
   
    num_mult = np.array(num_mult)
    # logger.info('')
    # logger.info('')
    # logger.info('=' * 80)
    # logger.info(f'#1 future: {len(np.where(num_mult == 1)[0])}/{pd.shape[0]}')
    # logger.info(f'#<10 future: {len(np.where(num_mult < 10)[0])}/{pd.shape[0]}')
    return traj_gt_arr

def get_prediction(data, model, sample_num, num_seeds=1, concat_hist=True):
    # 1 * total_len * num_key * 3
    dct_m, i_dct_m = get_dct_matrix(cfg.t_his+cfg.t_pred)
    dct_m = torch.from_numpy(dct_m).float().to(device)
    i_dct_m = torch.from_numpy(i_dct_m).float().to(device)
    traj_np = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)
    # 1 * total_len * ((num_key-1)*3)
    traj = tensor(traj_np, device=device, dtype=dtype).permute(1, 0, 2).contiguous()
    # total_len * 1 * ((num_key-1)*3)
    X = traj[:t_his]
    Y_gt = traj[t_his:]
    X = X.repeat((1, sample_num * num_seeds, 1))
    Y_gt = Y_gt.repeat((1, sample_num * num_seeds, 1))
    
    print('warm up ... \n')
    runtime_cost_list = []
    for _ in range(1000):
        start = time.time()
        Y, _, _ = model(X)
        torch.cuda.synchronize()
        end = time.time()
        print('Time:{}ms'.format((end-start)*1000))
        runtime_cost_list.append((end-start)*1000)
    runtime_cost = np.array(runtime_cost_list)
    runtime_cost = runtime_cost[1:]
    runtime_mean, runtime_std = np.mean(runtime_cost), np.std(runtime_cost)
    print(runtime_mean)
    print(runtime_std)
    
    Y, mu, logvar = model(X)
    
    if concat_hist:
        X = X.unsqueeze(2).repeat(1, sample_num * num_seeds, cfg.nk, 1)
        Y = Y[t_his:].unsqueeze(1)
        Y = torch.cat((X, Y), dim=0)
    # total_len * batch_size * feature_size
    Y = Y.squeeze(1).permute(1, 0, 2).contiguous().cpu().numpy()
    # batch_size * total_len * feature_size
    if Y.shape[0] > 1:
        Y = Y.reshape(-1, cfg.nk * sample_num, Y.shape[-2], Y.shape[-1])
    else:
        Y = Y[None, ...]
    return Y


def test(model, epoch):
    stats_func = {'Diversity': compute_diversity, 'AMSE': compute_amse, 'FMSE': compute_fmse, 'ADE': compute_ade,
                  'FDE': compute_fde, 'MMADE': compute_mmade, 'MMFDE': compute_mmfde, 'MPJPE': mpjpe_error}
    stats_names = list(stats_func.keys())
    stats_names.extend(['ADE_stat', 'FDE_stat', 'MMADE_stat', 'MMFDE_stat', 'MPJPE_stat'])
    stats_meter = {x: AverageMeter() for x in stats_names}

    data_gen = dataset_test.iter_generator(step=cfg.t_his)
    num_samples = 0
    num_seeds = 1
    
    for i, (data, _) in tqdm(enumerate(data_gen)):
        if args.mode == 'train' and (i >= 500 and (epoch + 1) % 50 != 0 and (epoch + 1) < cfg.num_epoch - 100):
            break
        num_samples += 1
        gt = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)[:, t_his:, :]
        gt_multi = traj_gt_arr[i]
        if gt_multi.shape[0] == 1:
            continue
    
        pred = get_prediction(data, model, sample_num=1, num_seeds=num_seeds, concat_hist=False)
        # pred = pred[:, (5, 15, 25, 35, 45)]
        pred = pred[:,:,t_his:,:]
        for stats in stats_names[:8]:
            val = 0
            branches = 0
            for pred_i in pred:
                # sample_num * total_len * ((num_key-1)*3), 1 * total_len * ((num_key-1)*3)
                v = stats_func[stats](pred_i, gt, gt_multi)
                val += v[0] / num_seeds
                if stats_func[stats](pred_i, gt, gt_multi)[1] is not None:
                    branches += v[1] / num_seeds
            stats_meter[stats].update(val)
            if type(branches) is not int:
                stats_meter[stats + '_stat'].update(branches)
    # logger.info('=' * 80)
    # for stats in stats_names:
    #     str_stats = f'Total {stats}: ' + f'{stats_meter[stats].avg}'
    #     logger.info(str_stats)
    # logger.info('=' * 80)
    return


def visualize():
    def denomarlize(*data):
        out = []
        for x in data:
            x = x * dataset.std + dataset.mean
            out.append(x)
        return out

    def post_process(pred, data):
        pred = pred.reshape(pred.shape[0], pred.shape[1], -1, 3)
        if cfg.normalize_data:
            pred = denomarlize(pred)
        pred = np.concatenate((np.tile(data[..., :1, :], (pred.shape[0], 1, 1, 1)), pred), axis=2)
        pred[..., :1, :] = 0
        return pred

    def pose_generator():
        while True:
            data, data_multimodal, action = dataset_test.sample(n_modality=10)
            gt = data[0].copy()
            gt[:, :1, :] = 0

            poses = {'action': action, 'context': gt, 'gt': gt}
            with torch.no_grad():
                pred = get_prediction(data, model, 1)[0]
                pred = post_process(pred, data)
                for i in range(pred.shape[0]):
                    poses[f'{i}'] = pred[i]
            yield poses

    pose_gen = pose_generator()
    for i in tqdm(range(args.n_viz)):
        render_animation(dataset.skeleton, pose_gen, cfg.t_his, ncol=12, output='./results/{}/results/'.format(args.cfg), index_i=i)


def visualization(model, epoch, action='all'):
    from visualization.vis_pose import plt_row_mixtures
    from visualization.vis_skeleton import VisSkeleton

    vis_skeleton = VisSkeleton(parents=[-1, 0, 1, 2, 3, 1, 5, 6, 0, 8, 9, 0, 11, 12, 1],
                                joints_left=[2, 3, 4, 8, 9, 10],
                                joints_right=[5, 6, 7, 11, 12, 13])
    
    save_subdir = action
    save_dir = os.path.join(os.getcwd(), 'output/imgs/HumanEva', save_subdir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("save_dir:" + str(save_dir))
    
    data_gen = dataset_test.iter_generator(step=cfg.t_his)
    for idx, (data, _) in enumerate(data_gen):
        
        gt = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)[:, t_his:, :]
        gz = np.zeros(shape=(gt.shape[0], gt.shape[1], 45))
        gz[..., 3:] = gt
        gz = np.reshape(gz, newshape=(gz.shape[0], gz.shape[1], 15, 3))
        x_pred = gz[:, [11, 23, 35, 47, 59]]  

        pred = get_prediction(data, model, sample_num=1, num_seeds=1, concat_hist=False)
        pz = np.zeros(shape=(pred.shape[0], pred.shape[1], pred.shape[2], 45))
        pz[..., 3:] = pred
        pz = np.reshape(pz, newshape=(pz.shape[0], pz.shape[1], pz.shape[2], 15, 3))
        pz = pz[:, (0,5,10,15,20,25,30,35,40,45), 15:]
        y = pz[:, :, [11, 23, 35, 47, 59]]
        y = np.swapaxes(y, 1, 2)      

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
                save_name = 'SLD_' + str(idx) + '_mix'
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
                save_name = 'SLD_' + str(idx)
            )
    return


def CMD_test(model, epoch):
    idx_to_class = ['directions', 'discussion', 'eating', 'greeting', 'phoning', \
                    'posing', 'purchases', 'sitting', 'sittingdown', 'smoking',  \
                    'photo', 'waiting', 'walking', 'walkdog', 'walktogether']
    mean_motion_per_class = [0.004528946212615328, 0.005068199383505345, 0.003978791804673771,  0.005921345536787865,   0.003595039379111546, 
                            0.004192961478268034, 0.005664689143238568, 0.0024945400286369122, 0.003543066357658834,   0.0035990843311130487, 
                            0.004356865838457266, 0.004219841185066826, 0.007528046315984569,  0.00007054820734533077, 0.006751761745020258]  

    def CMD(val_per_frame, val_ref):
        T = len(val_per_frame) + 1
        return np.sum([(T - t) * np.abs(val_per_frame[t-1] - val_ref) for t in range(1, T)])

    def CMD_helper(pred):
        pred_flat = pred   # shape: [batch, num_s, t_pred, joint, 3]
        # M = (torch.linalg.norm(pred_flat[:,:,1:] - pred_flat[:,:,:-1], axis=-1)).mean(axis=1).mean(axis=-1)    
        M = np.linalg.norm(pred_flat[:,:,1:] - pred_flat[:,:,:-1], axis=-1).mean(axis=1).mean(axis=-1) 
        return M

    def CMD_pose(data, label):
        ret = 0
        # CMD weighted by class
        for i, (name, class_val_ref) in enumerate(zip(idx_to_class, mean_motion_per_class)):
            mask = label == name
            if mask.sum() == 0:
                continue
            motion_data_mean = data[mask].mean(axis=0)
            ret += CMD(motion_data_mean, class_val_ref) * (mask.sum() / label.shape[0])
        return ret

    data_gen = dataset_test.iter_generator(step=cfg.t_his, afg=True)
    num_samples = 0
    num_seeds = 1

    M_list, label_list = [], []
    # data_shape: (1, 125, 17, 3)
    for data, _, action in data_gen:
    # for i, (data, _, action) in tqdm(enumerate(data_gen)):
        action = str.lower(re.sub(r'[0-9]+', '', action))
        action = re.sub(" ", "", action)
        
        if args.mode == 'train' and (i >= 500 and (epoch + 1) % 50 != 0 and (epoch + 1) < cfg.num_epoch - 100):
            break
        num_samples += 1
        gt = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)[:, t_his:, :]
        # gt_multi = traj_gt_arr[i]
        # if gt_multi.shape[0] == 1:
        #     continue
        
        # pred_shape: (1, 50, 125, 48)
        pred = get_prediction(data, model, sample_num=1, num_seeds=num_seeds, concat_hist=False)
        pred = pred[:,:,t_his:,:]
        pred = pred.reshape(1, 50, 100, 16, 3)
        pred = pred[:, (5, 15, 25, 35, 45)]
        
        M = CMD_helper(pred)
        M_list.append(M)
        label_list.append(action)

    M_all = np.concatenate(M_list, 0)
    label_all = np.array(label_list)
    
    cmd_score = CMD_pose(M_all, label_all) 
    print(cmd_score)
    return


def CMD_test_eva(model, epoch):
    idx_to_class = ['Box', 'Gestures', 'Jog', 'ThrowCatch', 'Walking']
    mean_motion_per_class = [0.010139551, 0.0021507503, 0.010850595,  0.004398426,   0.006771291]  

    def CMD(val_per_frame, val_ref):
        T = len(val_per_frame) + 1
        return np.sum([(T - t) * np.abs(val_per_frame[t-1] - val_ref) for t in range(1, T)])

    def CMD_helper(pred):
        pred_flat = pred   # shape: [batch, num_s, t_pred, joint, 3] 
        M = np.linalg.norm(pred_flat[:,:,1:] - pred_flat[:,:,:-1], axis=-1).mean(axis=1).mean(axis=-1) 
        return M

    def CMD_pose(data, label):
        ret = 0
        # CMD weighted by class
        for i, (name, class_val_ref) in enumerate(zip(idx_to_class, mean_motion_per_class)):
            mask = label == name
            if mask.sum() == 0:
                continue
            motion_data_mean = data[mask].mean(axis=0)
            ret += CMD(motion_data_mean, class_val_ref) * (mask.sum() / label.shape[0])
        return ret

    data_gen = dataset_test.iter_generator(step=cfg.t_his, afg=True)
    num_samples = 0
    num_seeds = 1

    M_list, label_list = [], []
    for data, _, action in data_gen:        
        if args.mode == 'train' and (i >= 500 and (epoch + 1) % 50 != 0 and (epoch + 1) < cfg.num_epoch - 100):
            break
        num_samples += 1
        gt = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)[:, t_his:, :]

        pred = get_prediction(data, model, sample_num=1, num_seeds=num_seeds, concat_hist=False)
        pred = pred[:,:,t_his:,:]
        pred = pred.reshape(1, 50, 60, 14, 3)
        pred = pred[:, (5, 15, 25, 35, 45)]
        
        M = CMD_helper(pred)
        M_list.append(M)
        label_list.append(action)

    M_all = np.concatenate(M_list, 0)
    label_all = np.array(label_list)
    
    cmd_score = CMD_pose(M_all, label_all) 
    print(cmd_score)
    return

    
def FID_test(model, epoch, classifier):
    data_gen = dataset_test.iter_generator(step=cfg.t_his, afg=True)
    num_samples = 0
    num_seeds = 1

    pred_act_list, gt_act_list = [], []
    # data_shape: (1, 125, 17, 3)
    for i, (data, _, action) in tqdm(enumerate(data_gen)):
        # action = str.lower(re.sub(r'[0-9]+', '', action))
        # action = re.sub(" ", "", action)
        
        if args.mode == 'train' and (i >= 500 and (epoch + 1) % 50 != 0 and (epoch + 1) < cfg.num_epoch - 100):
            break
        num_samples += 1
        gt = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)[:, t_his:, :]
        gt = np.repeat(gt, 50, axis=0)
        gt = np.swapaxes(gt, 1, 2)
        gt = torch.tensor(gt, device=device)
        
        ''' ### h36m
        # pred_shape: (1, 50, 125, 48)
        pred = get_prediction(data, model, sample_num=1, num_seeds=num_seeds, concat_hist=False)
        pred = pred[:,:,t_his:,:]
        pred = np.swapaxes(pred, -2, -1)
        pred = pred.reshape(50, 48, 100)
        # pred = pred[(5, 15, 25, 35, 45), ...]
        pred = torch.tensor(pred, device=device)
        '''

        # pred_shape: (1, 50, 125, 48)
        pred = get_prediction(data, model, sample_num=1, num_seeds=num_seeds, concat_hist=False)
        pred = pred[:,:,t_his:,:]
        pred = np.swapaxes(pred, -2, -1)
        pred = pred.reshape(50, 42, 60)
        pred = pred[(5, 15, 25, 35, 45), ...]
        pred = torch.tensor(pred, device=device)
        
        pred_activations = classifier.get_fid_features(motion_sequence=pred).cpu().data.numpy()
        gt_activations   = classifier.get_fid_features(motion_sequence=gt).cpu().data.numpy()
        
        pred_act_list.append(pred_activations)
        gt_act_list.append(gt_activations)
        
    results_fid = fid(np.concatenate(pred_act_list, 0), np.concatenate(gt_act_list, 0))
    print(results_fid)


def KDE_test(model, epoch):
    data_gen = dataset_test.iter_generator(step=cfg.t_his)
    num_samples = 0
    num_seeds = 1
    
    kde_list = []
    for i, (data, _) in tqdm(enumerate(data_gen)):
        torch.cuda.empty_cache()
        if args.mode == 'train' and (i >= 500 and (epoch + 1) % 50 != 0 and (epoch + 1) < cfg.num_epoch - 100):
            break
        num_samples += 1
        gt = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)[:, t_his:, :]
        gt = np.reshape(gt, (gt.shape[0], gt.shape[1], -1, 3))

        pred_thousand = []
        
        for j in range(20):
            pred = get_prediction(data, model, sample_num=1, num_seeds=num_seeds, concat_hist=False)
            pred = pred[:,:,t_his:,:]  
            pred_thousand.append(pred)
        pred_thousand = np.concatenate(pred_thousand, axis=1)
        pred = pred_thousand
        pred = np.reshape(pred, (pred.shape[0], pred.shape[1], pred.shape[2], -1, 3))
        # pred = get_prediction(data, model, sample_num=1, num_seeds=num_seeds, concat_hist=False)
        # pred = pred[:,:,t_his:,:]  
        
        kde_list.append(kde(y=gt, y_pred=pred))

    kde_ll = torch.cat(kde_list, dim=0)
    kde_ll = torch.mean(kde_ll, dim=0)
    kde_ll_np = kde_ll.to('cpu').numpy()
    print(kde_ll_np)
    return



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',
                        default='humaneva')
    parser.add_argument('--mode', default='CMD')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--iter', type=int, default=500)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--n_pre', type=int, default=8)
    parser.add_argument('--n_his', type=int, default=5)
    parser.add_argument('--n_viz', type=int, default=100)
    parser.add_argument('--num_coupling_layer', type=int, default=4)
    parser.add_argument('--multimodal_threshold', type=float, default=0.5)
    args = parser.parse_args()
    
    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    
    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    
    cfg = Config(f'{args.cfg}', test=args.test)
    # tb_logger = SummaryWriter(cfg.tb_dir) if args.mode == 'train' else None
    # logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))
        
    """parameter"""
    mode = args.mode
    nz = cfg.nz
    t_his = cfg.t_his
    t_pred = cfg.t_pred
    cfg.n_his = args.n_his
    if 'n_pre' not in cfg.specs.keys():
        cfg.n_pre = args.n_pre
    else:
        cfg.n_pre = cfg.specs['n_pre']
    cfg.num_coupling_layer = args.num_coupling_layer
    # cfg.nz = args.nz
    """data"""
    if 'actions' in cfg.specs.keys():
        act = cfg.specs['actions']
    else:
        act = 'all'
    dataset_cls = DatasetH36M if cfg.dataset == 'h36m' else DatasetHumanEva
    dataset = dataset_cls('train', t_his, t_pred, actions=act, use_vel=cfg.use_vel,
                          multimodal_path=cfg.specs[
                              'multimodal_path'] if 'multimodal_path' in cfg.specs.keys() else None,
                          data_candi_path=cfg.specs[
                              'data_candi_path'] if 'data_candi_path' in cfg.specs.keys() else None)
    dataset_test = dataset_cls('test', t_his, t_pred, actions=act, use_vel=cfg.use_vel,
                               multimodal_path=cfg.specs[
                                   'multimodal_path'] if 'multimodal_path' in cfg.specs.keys() else None,
                               data_candi_path=cfg.specs[
                                   'data_candi_path'] if 'data_candi_path' in cfg.specs.keys() else None)
    if cfg.normalize_data:
        dataset.normalize_data()
        dataset_test.normalize_data(dataset.mean, dataset.std)
    traj_gt_arr = get_multimodal_gt(dataset_test)
    """model"""
    model, pose_prior = get_model(cfg, dataset, cfg.dataset)
    
    model.float()
    pose_prior.float()
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    
    scheduler = get_scheduler(optimizer, policy='lambda', nepoch_fix=cfg.num_epoch_fix, nepoch=cfg.num_epoch)

    cp_path = 'results/h36m_nf/models/0025.p' if cfg.dataset == 'h36m' else 'results/humaneva_nf/models/0025.p'
    print('loading model from checkpoint: %s' % cp_path)
    model_cp = pickle.load(open(cp_path, "rb"))
    pose_prior.load_state_dict(model_cp['model_dict'])
    pose_prior.to(device)

    valid_ang = pickle.load(open('./data/h36m_valid_angle.p', "rb")) if cfg.dataset == 'h36m' else pickle.load(
        open('./data/humaneva_valid_angle.p', "rb"))
    if args.iter > 0:
        cp_path = cfg.model_path % args.iter
        print('loading model from checkpoint: %s' % cp_path)
        model_cp = pickle.load(open(cp_path, "rb"))
        model.load_state_dict(model_cp['model_dict'])
        print("load done")
        
    if mode == 'train':
        model.to(device)
        overall_iter = 0
        stats = torch.zeros(cfg.nk)
        model.train()

        for i in range(args.iter, cfg.num_epoch):
            stats = train(i, stats) 
            if cfg.save_model_interval > 0 and (i + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    test(model, i) 
                model.train()
                with to_cpu(model):
                    cp_path = cfg.model_path % (i + 1)
                    model_cp = {'model_dict': model.state_dict(), 'meta': {'std': dataset.std, 'mean': dataset.mean}}
                    
                    pickle.dump(model_cp, open(cp_path, 'wb'))
                       
    elif mode == 'test':
        model.to(device)
        model.eval()
        with torch.no_grad():
            test(model, args.iter) 
    
    elif mode == 'CMD':
        model.to(device)
        model.eval()
        with torch.no_grad():
            if cfg.dataset == 'h36m':
                CMD_test(model,args.iter)
            else:
                CMD_test_eva(model,args.iter)     

    elif mode == 'FID':
        classifier = classifier_fid_factory(device) if cfg.dataset == 'h36m' else classifier_fid_humaneva_factory(device)
        model.to(device)
        model.eval()
        with torch.no_grad():
            FID_test(model, args.iter, classifier)   

    elif mode == 'vis':
        model.to(device)
        model.eval()
        with torch.no_grad():
            # visualize()
            visualization(model, args.iter)
    
    elif mode == 'KDE':
        model.to(device)
        model.eval()
        with torch.no_grad():
            KDE_test(model, args.iter)
