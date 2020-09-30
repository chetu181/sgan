import argparse
import os
import torch

from attrdict import AttrDict

from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path

import matplotlib
from matplotlib import pyplot as plt
# from matplotlib.pyplot import savefig
import numpy as np
# matplotlib.use('TKAgg')


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--trajectories_to_show_per_batch', default=5   , type=int)
parser.add_argument('--seed', default=10, type=int)
parser.add_argument('--dset_type', default='test', type=str)

args = parser.parse_args()
global_model_path = args.model_path
np.random.seed(args.seed)
torch.manual_seed(args.seed)
print("global_model_path", global_model_path)
def plottraj(ax, trs, label, col='r', alpha = 0.3): # trs is 3 dimensional, <pred_len(step1,step2 etc)>, <batch_size>, <2 for x,y coord>
    print("\nplotting one trajectory")
    print(label,trs.shape)
    # trs = trajectory.cpu().detach().numpy()
    trs = np.transpose(trs, (1, 2, 0))
    print("trs.shape", trs.shape)
    # print("trs", trs)
    
    scatterlength = 1
    for i, tr in enumerate(trs[:args.trajectories_to_show_per_batch]):
        if(i==0):
            ax.text(tr[0][0], tr[1][0],str(i))
            ax.plot(tr[0], tr[1], color=col, label=label, alpha=alpha)
            ax.scatter(tr[0][:scatterlength], tr[1][:scatterlength], color=col, alpha=alpha)
        else:
            ax.text(tr[0][0], tr[1][0],str(i))
            ax.plot(tr[0], tr[1], color=col, alpha=alpha)
            ax.scatter(tr[0][:scatterlength], tr[1][:scatterlength], color=col, alpha=alpha)


def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    #print("args for the checkpoint", args.model_path, ":")
    print(args)
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm,
        use_gpu=args.use_gpu)
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator


def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_


def evaluate(args, loader, generator, num_samples):
    ade_outer, fde_outer = [], []
    total_traj = 0
    fig, ax = plt.subplots(num='3d time-surface',figsize=(16,12))
    
    topleft = (103.80, 1.21)
    bottomright = (103.86, 1.16)

    def plotrectangle(ax, topleft, bottomright):
        x1 = topleft[0]
        y1 = topleft[1]
        x2 = bottomright[0]
        y2 = bottomright[1]
        bordercol = 'black'
        borderalpha = 0.6
        ax.plot([x1, x2], [y1, y1], color=bordercol, alpha=borderalpha)
        ax.plot([x1, x2], [y2, y2], color=bordercol, alpha=borderalpha)
        ax.plot([x1, x1], [y1, y2],color=bordercol, alpha=borderalpha)
        ax.plot([x2, x2], [y1, y2],color=bordercol, alpha=borderalpha)

    plotrectangle(ax, topleft, bottomright)

    cols = ['b','b','b','b']
    colid = 0
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch

            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)
            np_obs_traj = obs_traj.cpu().detach().numpy()
            np_pred_traj_gt = pred_traj_gt.cpu().detach().numpy()
            print("np_pred_traj_gt.shape", np_pred_traj_gt.shape)
            print("np_obs_traj[-1,:,:].shape", np_obs_traj[-1:,:,:].shape)
            np_pred_traj_gt = np.concatenate((np_obs_traj[-1:,:,:], np_pred_traj_gt),axis=0)
            # np_pred_traj_gt reshape(,33,2)
            plottraj(ax, np_obs_traj , 'obs_traj', cols[colid], alpha=1.0)
            colid = (colid + 1) % 4
            plottraj(ax, np_pred_traj_gt, 'pred_traj_gt', 'g', alpha=1.0)
            
            for _ in range(num_samples):
                pred_traj_fake_rel = generator(
                    obs_traj, obs_traj_rel, seq_start_end
                )
                pred_traj_fake = relative_to_abs(
                    pred_traj_fake_rel, obs_traj[-1]
                )
                ade.append(displacement_error(
                    pred_traj_fake, pred_traj_gt, mode='raw'
                ))
                fde.append(final_displacement_error(
                    pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
                ))
                np_pred_traj_fake = pred_traj_fake.cpu().detach().numpy()
                np_pred_traj_fake = np.concatenate((np_obs_traj[-1:,:,:], np_pred_traj_fake),axis=0)
                plottraj(ax, np_pred_traj_fake,'pred_traj_fake','r')
            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)
            break
        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / (total_traj)
        print("ade ", ade)
        plt.legend(bbox_to_anchor=(1.0, 1.1), loc='upper left', title="legend", borderaxespad=0.)
        # ax.autoscale(False)
        ax.set_aspect('equal')
        # plt.show()
        plt.savefig('img'+global_model_path+'.png')
        return ade, fde


def main(args):
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]

    for path in paths:
        checkpoint = torch.load(path)
        print("args for model_path", args.model_path, ":")
        generator = get_generator(checkpoint)
        _args = AttrDict(checkpoint['args'])
        path = get_dset_path(_args.dataset_name, args.dset_type)
        _, loader = data_loader(_args, path)
        ade, fde = evaluate(_args, loader, generator, args.num_samples)
        if "mtm" in args.model_path:
            ade, fde = ade * 110.0 , fde * 110.0 # converting from lat-lon to kms
            print("ade and fde in kms")
        print('Dataset: {}, Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}'.format(
            _args.dataset_name, _args.pred_len, ade, fde))


if __name__ == '__main__':
    
    main(args)
