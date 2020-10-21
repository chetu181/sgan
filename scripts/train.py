import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import gc
import logging
import os
import sys
import time

from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

from sgan.data.loader import data_loader
from sgan.losses import gan_g_loss, gan_d_loss, l2_loss
from sgan.losses import displacement_error, final_displacement_error

from sgan.models import TrajectoryGenerator, TrajectoryDiscriminator
from sgan.utils import int_tuple, bool_flag, get_total_norm
from sgan.utils import relative_to_abs, get_dset_path

import tensorflow as tf
import datetime
from torch import autograd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

torch.set_printoptions(edgeitems=1000)
torch.backends.cudnn.benchmark = True
torch.manual_seed(13)

parser = argparse.ArgumentParser()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Dataset options
parser.add_argument('--dataset_name', default='zara1', type=str)
parser.add_argument('--delim', default=' ')
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--pred_len', default=8, type=int)
parser.add_argument('--skip', default=1, type=int)

# Optimization
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_iterations', default=10000, type=int)
parser.add_argument('--num_epochs', default=200, type=int)

# Model Options
parser.add_argument('--embedding_dim', default=64, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--dropout', default=0, type=float)
parser.add_argument('--batch_norm', default=0, type=bool_flag)
parser.add_argument('--mlp_dim', default=1024, type=int)

# Generator Options
parser.add_argument('--encoder_h_dim_g', default=64, type=int)
parser.add_argument('--decoder_h_dim_g', default=128, type=int)
parser.add_argument('--noise_dim', default=None, type=int_tuple)
parser.add_argument('--noise_type', default='gaussian')
parser.add_argument('--noise_mix_type', default='ped')
parser.add_argument('--clipping_threshold_g', default=0, type=float)
parser.add_argument('--g_learning_rate', default=5e-4, type=float)
parser.add_argument('--g_steps', default=1, type=int)

# Pooling Options
parser.add_argument('--pooling_type', default='pool_net')
parser.add_argument('--pool_every_timestep', default=1, type=bool_flag)

# Pool Net Option
parser.add_argument('--bottleneck_dim', default=1024, type=int)

# Social Pooling Options
parser.add_argument('--neighborhood_size', default=2.0, type=float)
parser.add_argument('--grid_size', default=8, type=int)

# Discriminator Options
parser.add_argument('--d_type', default='local', type=str)
parser.add_argument('--encoder_h_dim_d', default=64, type=int)
parser.add_argument('--d_learning_rate', default=5e-4, type=float)
parser.add_argument('--d_steps', default=2, type=int)
parser.add_argument('--clipping_threshold_d', default=0, type=float)
parser.add_argument('--use_discriminator', default=True, type=bool)
# Loss Options
parser.add_argument('--l2_loss_weight', default=0, type=float)
parser.add_argument('--cosine_loss_weight', default=0, type=float)
parser.add_argument('--curvature_loss_weight', default=0, type=float)
parser.add_argument('--best_k', default=1, type=int)

# Output
parser.add_argument('--output_dir', default=os.getcwd())
parser.add_argument('--print_every', default=5, type=int)
parser.add_argument('--checkpoint_every', default=100, type=int)
parser.add_argument('--checkpoint_name', default='checkpoint')
parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--restore_from_checkpoint', default=1, type=int)
parser.add_argument('--num_samples_check', default=5000, type=int)

# Misc
parser.add_argument('--use_gpu', default=False, type=bool)
parser.add_argument('--use_tboard', default=False, type=bool)
parser.add_argument('--timing', default=0, type=int)
parser.add_argument('--gpu_num', default="0", type=str)
args = parser.parse_args()

use_tboard = args.use_tboard
# Tensorboard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = None
#writer.add_graph(tf.get_default_graph())

# start a session (this should be done quite late)
if(use_tboard):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

placeholders={}
summary_tensors={}

def add_summary(legend, value, step):
    if(legend not in placeholders): # create summary tensor if not present.
        placeholders[legend] = tf.placeholder(dtype=tf.float32)
        summary_tensors[legend] = tf.summary.scalar(
            name=legend, tensor=placeholders[legend])
    summary = sess.run(summary_tensors[legend], feed_dict={placeholders[legend]:value})
    writer.add_summary(summary, step)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)


def get_dtypes(args):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if args.use_gpu:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    train_path = get_dset_path(args.dataset_name, 'train')
    val_path = get_dset_path(args.dataset_name, 'val')

    long_dtype, float_dtype = get_dtypes(args)

    logger.info("Initializing train dataset")
    train_dset, train_loader = data_loader(args, train_path)
    logger.info("Initializing val dataset")
    _, val_loader = data_loader(args, val_path)

    iterations_per_epoch = len(train_dset) / args.batch_size / args.d_steps
    print(" len(train_dset)", len(train_dset))
    print("args.batch_size",  args.batch_size)
    print("args.d_steps", args.d_steps)
    print("iterations_per_epoch", iterations_per_epoch)
    
    if args.num_epochs:
        args.num_iterations = int(iterations_per_epoch * args.num_epochs)

    logger.info(
        'There are {} iterations per epoch'.format(iterations_per_epoch)
    )

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

    generator.apply(init_weights)
    generator.type(float_dtype).train()
    logger.info('Here is the generator:')
    # logger.info(generator)

    discriminator = TrajectoryDiscriminator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        h_dim=args.encoder_h_dim_d,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_norm=args.batch_norm,
        d_type=args.d_type,
        use_gpu=args.use_gpu)

    discriminator.apply(init_weights)
    discriminator.type(float_dtype).train()
    logger.info('Here is the discriminator:')
    # logger.info(discriminator)

    g_loss_fn = gan_g_loss
    d_loss_fn = gan_d_loss
    
    optimizer_g = optim.Adam(generator.parameters(), lr=args.g_learning_rate)
    optimizer_d = optim.Adam(
        discriminator.parameters(), lr=args.d_learning_rate
    )

    # Maybe restore from checkpoint
    restore_path = None
    if args.checkpoint_start_from is not None:
        restore_path = args.checkpoint_start_fromf
    elif args.restore_from_checkpoint == 1:
        restore_path = os.path.join(args.output_dir,
                                    '%s_with_model.pt' % args.checkpoint_name)

    if restore_path is not None and os.path.isfile(restore_path):
        logger.info('Restoring from checkpoint {}'.format(restore_path))
        checkpoint = torch.load(restore_path)
        generator.load_state_dict(checkpoint['g_state'])
        discriminator.load_state_dict(checkpoint['d_state'])
        optimizer_g.load_state_dict(checkpoint['g_optim_state'])
        optimizer_d.load_state_dict(checkpoint['d_optim_state'])
        t = checkpoint['counters']['t']
        epoch = checkpoint['counters']['epoch']
        checkpoint['restore_ts'].append(t)
    else:
        # Starting from scratch, so initialize checkpoint data structure
        t, epoch = 0, 0
        checkpoint = {
            'args': args.__dict__,
            'G_losses': defaultdict(list),
            'D_losses': defaultdict(list),
            'losses_ts': [],
            'metrics_val': defaultdict(list),
            'metrics_train': defaultdict(list),
            'sample_ts': [],
            'restore_ts': [],
            'norm_g': [],
            'norm_d': [],
            'counters': {
                't': None,
                'epoch': None,
            },
            'g_state': None,
            'g_optim_state': None,
            'd_state': None,
            'd_optim_state': None,
            'g_best_state': None,
            'd_best_state': None,
            'best_t': None,
            'g_best_nl_state': None,
            'd_best_state_nl': None,
            'best_t_nl': None,
        }
    t0 = None
    while t < args.num_iterations:
        #add_summary('t_iter', t, t)
        print("t = ", t)
        # print("generator weights: ", [x.data for x in generator.decoder.parameters()])
        gc.collect()
        d_steps_left = args.d_steps
        g_steps_left = args.g_steps
        epoch += 1
        logger.info('Starting epoch {}'.format(epoch))
        for batch in train_loader:
            if args.timing == 1:
                torch.cuda.synchronize()
                t1 = time.time()

            # Decide whether to use the batch for stepping on discriminator or
            # generator; an iteration consists of args.d_steps steps on the
            # discriminator followed by args.g_steps steps on the generator.
            if d_steps_left > 0:
                step_type = 'd'
                losses_d = discriminator_step(args, batch, generator,
                                              discriminator, d_loss_fn,
                                              optimizer_d)
                checkpoint['norm_d'].append(
                    get_total_norm(discriminator.parameters()))
                d_steps_left -= 1
            elif g_steps_left > 0:
                step_type = 'g'
                print("get step t = ", t)
                losses_g = generator_step(args, batch, generator,
                                          discriminator, g_loss_fn,
                                          optimizer_g)
                checkpoint['norm_g'].append(
                    get_total_norm(generator.parameters())
                )
                g_steps_left -= 1

            if args.timing == 1:
                torch.cuda.synchronize()
                t2 = time.time()
                logger.info('{} step took {}'.format(step_type, t2 - t1))

            # Skip the rest if we are not at the end of an iteration
            if d_steps_left > 0 or g_steps_left > 0:
                continue

            if args.timing == 1:
                if t0 is not None:
                    logger.info('Interation {} took {}'.format(
                        t - 1, time.time() - t0
                    ))
                t0 = time.time()

            # Maybe save loss
            if t % args.print_every == 0:
                logger.info('t = {} / {}'.format(t + 1, args.num_iterations))
                for k, v in sorted(losses_d.items()):
                    logger.info('  [D] {}: {:.3f}'.format(k, v))
                    checkpoint['D_losses'][k].append(v)
                    if(use_tboard): add_summary('Dis_'+k, v, t)
                for k, v in sorted(losses_g.items()):
                    logger.info('  [G] {}: {:.3f}'.format(k, v))
                    checkpoint['G_losses'][k].append(v)
                    if(use_tboard): add_summary('Gen_'+k, v, t)
                checkpoint['losses_ts'].append(t)

            # Maybe save a checkpoint
            if t > 0 and t % args.checkpoint_every == 0:
                checkpoint['counters']['t'] = t
                checkpoint['counters']['epoch'] = epoch
                checkpoint['sample_ts'].append(t)

                # Check stats on the validation set
                logger.info('Checking stats on val ...')
                metrics_val = check_accuracy(
                    args, val_loader, generator, discriminator, d_loss_fn
                )
                logger.info('Checking stats on train ...')
                metrics_train = check_accuracy(
                    args, train_loader, generator, discriminator,
                    d_loss_fn, limit=True
                )
                if(use_tboard):
                    add_summary('ade_train', metrics_train['ade'], t)
                    add_summary('ade_VAL', metrics_val['ade'], t)
                    add_summary('FDE_train', metrics_train['fde'], t)
                    add_summary('FDE_VAL', metrics_val['fde'], t)

                for k, v in sorted(metrics_val.items()):
                    logger.info('  [val] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_val'][k].append(v)
                for k, v in sorted(metrics_train.items()):
                    logger.info('  [train] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_train'][k].append(v)

                min_ade = min(checkpoint['metrics_val']['ade'])
                min_ade_nl = min(checkpoint['metrics_val']['ade_nl'])

                if metrics_val['ade'] == min_ade:
                    logger.info('New low for avg_disp_error')
                    checkpoint['best_t'] = t
                    checkpoint['g_best_state'] = generator.state_dict()
                    checkpoint['d_best_state'] = discriminator.state_dict()

                if metrics_val['ade_nl'] == min_ade_nl:
                    logger.info('New low for avg_disp_error_nl')
                    checkpoint['best_t_nl'] = t
                    checkpoint['g_best_nl_state'] = generator.state_dict()
                    checkpoint['d_best_nl_state'] = discriminator.state_dict()

                # Save another checkpoint with model weights and
                # optimizer state
                checkpoint['g_state'] = generator.state_dict()
                checkpoint['g_optim_state'] = optimizer_g.state_dict()
                checkpoint['d_state'] = discriminator.state_dict()
                checkpoint['d_optim_state'] = optimizer_d.state_dict()
                checkpoint_path = os.path.join(
                    args.output_dir, '%s_with_model.pt' % args.checkpoint_name
                )
                logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                torch.save(checkpoint, checkpoint_path)
                logger.info('Done.')

                # Save a checkpoint with no model weights by making a shallow
                # copy of the checkpoint excluding some items
                checkpoint_path = os.path.join(
                    args.output_dir, '%s_no_model.pt' % args.checkpoint_name)
                logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                key_blacklist = [
                    'g_state', 'd_state', 'g_best_state', 'g_best_nl_state',
                    'g_optim_state', 'd_optim_state', 'd_best_state',
                    'd_best_nl_state'
                ]
                small_checkpoint = {}
                for k, v in checkpoint.items():
                    if k not in key_blacklist:
                        small_checkpoint[k] = v
                torch.save(small_checkpoint, checkpoint_path)
                logger.info('Done.')

            t += 1
            d_steps_left = args.d_steps
            g_steps_left = args.g_steps
            if t >= args.num_iterations:
                break
# close tensorboard writer
#writer.flush()
#writer.close()



def discriminator_step(
    args, batch, generator, discriminator, d_loss_fn, optimizer_d
):
    if(args.use_gpu):
        batch = [tensor.cuda() for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
     loss_mask, seq_start_end) = batch
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)

    generator_out = generator(obs_traj, obs_traj_rel, seq_start_end)

    pred_traj_fake_rel = generator_out
    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

    traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
    traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

    scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
    scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)

    # Compute loss with optional gradient penalty
    data_loss = d_loss_fn(scores_real, scores_fake)
    losses['D_data_loss'] = data_loss.item()
    loss += data_loss
    losses['D_total_loss'] = loss.item()

    optimizer_d.zero_grad()
    loss.backward()
    if args.clipping_threshold_d > 0:
        nn.utils.clip_grad_norm_(discriminator.parameters(),
                                 args.clipping_threshold_d)
    optimizer_d.step()

    return losses


def generator_step(
    args, batch, generator, discriminator, g_loss_fn, optimizer_g
):
    if(args.use_gpu):
        batch = [tensor.cuda() for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
     loss_mask, seq_start_end) = batch
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)
    g_l2_loss_rel = []

    loss_mask = loss_mask[:, args.obs_len:]

    for _ in range(args.best_k):
        generator_out = generator(obs_traj, obs_traj_rel, seq_start_end)

        pred_traj_fake_rel = generator_out
        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

        if args.l2_loss_weight > 0:
            g_l2_loss_rel.append(args.l2_loss_weight * l2_loss(
                pred_traj_fake_rel,
                pred_traj_gt_rel,
                loss_mask,
                mode='raw'))

    g_l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
    if args.l2_loss_weight > 0:
        g_l2_loss_rel = torch.stack(g_l2_loss_rel, dim=1)
        for start, end in seq_start_end.data:
            _g_l2_loss_rel = g_l2_loss_rel[start:end]
            _g_l2_loss_rel = torch.sum(_g_l2_loss_rel, dim=0)
            _g_l2_loss_rel = torch.min(_g_l2_loss_rel) / torch.sum(
                loss_mask[start:end])
            g_l2_loss_sum_rel += _g_l2_loss_rel
        losses['G_l2_loss_rel'] = g_l2_loss_sum_rel.item()
        loss += g_l2_loss_sum_rel

    if args.cosine_loss_weight > 0: # TODO : what to do when curvature loss needs to be included?
        # need to obtain the curvature for all data points in the batch. And save their average to curvature_loss tensor..
        # given pred_traj_fake_rel, and last two inputs(may be not needed for first iteration). pred_traj_gt_rel(not needed)
        # print("COSINE: pred_traj_fake_rel.shape", pred_traj_fake_rel.shape)
        # print("pred_traj_fake_rel", pred_traj_fake_rel) # just use this tensor for now, then you can add other stuff later.
        #pred_traj_fake_rel_shifted = torch.roll(pred_traj_fake_rel, 1, [1])
        pred_traj_fake_rel_shifted = torch.cat( (pred_traj_fake_rel[1:, :, :],  pred_traj_fake_rel[:1, :, :]), dim=0) # simulating rolling
        # print("pred_traj_fake_rel_shifted", pred_traj_fake_rel_shifted)
        cosfunc = nn.CosineSimilarity(dim=2, eps=1e-6)
        similarity = cosfunc(pred_traj_fake_rel, pred_traj_fake_rel_shifted)
        # print("similarity", similarity)
        cosine_loss = torch.mean(similarity[1:,:])
        losses['G_cosine_loss'] = cosine_loss.item()
        print("cosine_loss", cosine_loss)
        loss += -args.cosine_loss_weight * cosine_loss
        # sys.exit()
    curv_debug = False
    print("loss before adding curvature", loss)
    if args.curvature_loss_weight > 0:
        print("CURVATURE: pred_traj_fake_rel.shape", pred_traj_fake_rel.shape)
        # print("pred_traj_fake_rel", pred_traj_fake_rel) # just use this tensor for now, then you can add other stuff later.
        dists = torch.norm(pred_traj_fake_rel,p=2, dim=2)
        # print("dists", dists)
        print("dists.shape", dists.shape)

        # get 3 sets of points aas, bbs and ccs corresponding to 3 subsequent points
        aas = torch.cat( (obs_traj[-2:,:,:], pred_traj_fake[:-2,:,:]) , dim=0)
        print("aas.dtype", aas.dtype)
        bbs = torch.cat( (obs_traj[-1:,:,:], pred_traj_fake[:-1,:,:]) , dim=0)
        ccs = pred_traj_fake
        # print("obs_traj[-3:,:3,:]", obs_traj[-3:,:3,:])
        if(curv_debug):
            print("aas.shape", aas.shape)
            print("bbs.shape", bbs.shape)
            print("ccs.shape", ccs.shape)               
            print("aas:\n", aas)
            print("bbs:\n", bbs)
            print("ccs:\n", ccs)    
        cside = torch.norm(aas-bbs ,p=2, dim=2) 
        bside = torch.norm(aas-ccs ,p=2, dim=2)
        aside = torch.norm(ccs-bbs ,p=2, dim=2)
        # print("cside.shape", cside.shape)
        s = (aside + bside + cside) / 2
        areas = torch.sqrt(s * (s-aside)* (s-bside)* (s-cside))

        if curv_debug:
            onne = torch.ones(areas.shape).cuda()
            zeero = torch.zeros(areas.shape).cuda()
            print("onne.shape", onne.shape)
            print("zeero.shape", zeero.shape)
            debug_ars = torch.where(areas<0.00001, onne, zeero)
            areas = torch.where(areas < 0.0001, zeero, areas)
            debug_ars_sum = torch.sum(debug_ars)
            print("debug_ars_sum: ", debug_ars_sum)
        # print("areas : ", areas)
        curvatures = areas/aside/bside/cside
        exp = (curvatures != curvatures)
        curvatures[exp] = 0
        # curvatures = curvatures * exp
        # print("curvatures.shape", curvatures.shape)
        # print("aside", aside)
        # print("bside", bside)
        # print("cside", cside)

        # print("areas", areas)
        # print("curvatures", curvatures)
        # curvatures[curvatures != curvatures] = 0
        curvature_loss = torch.mean(curvatures)
        # curvature_loss[curvature_lFoss != curvature_loss] = 0 #setting nan as zero
        print("curvature_loss", curvature_loss)
        losses['G_curvature_loss'] = curvature_loss.item()
        loss += args.curvature_loss_weight * curvature_loss
        # sys.exit()

    if(loss != loss):
        print("[id]" , "is where you see NaN first")
        print("loss", loss)
        print("curvature_loss Nan", curvature_loss)
        print("\n\n==================NAN================\n\n")
        nan_loc = 0
        nan_id = 7
        print("\n=====  curvatures[nan_loc,nan_id:nan_id+5]\n", curvatures[nan_loc,nan_id :nan_id+5])
        print("\n=====  areas[nan_loc,nan_id:nan_id+5]\n", areas[nan_loc,nan_id :nan_id+5])
        print("\n=====  aside[nan_loc,nan_id:nan_id+5]\n", aside[nan_loc,nan_id :nan_id+5])
        print("\n=====  bside[nan_loc,nan_id:nan_id+5]\n", bside[nan_loc,nan_id :nan_id+5])
        print("\n=====  cside[nan_loc,nan_id:nan_id+5]\n", cside[nan_loc,nan_id :nan_id+5])
        
        
        print("aas.shape", aas.shape)
        print("bbs.shape", bbs.shape)
        print("ccs.shape", ccs.shape)               
        print("aas:\n", aas[nan_loc,nan_id :nan_id+5])
        print("bbs:\n", bbs[nan_loc,nan_id :nan_id+5])
        print("ccs:\n", ccs[nan_loc,nan_id :nan_id+5])

        exp = (curvatures == curvatures)
        print("obs_traj", obs_traj[:,6:11,:])
        print("pred_traj_fake", pred_traj_fake[:,6:11,:])
        print("obs_traj.shape", obs_traj.shape)
        print("pred_traj_fake.shape", pred_traj_fake.shape)
        print ("exp", exp[:,6:11])
        # print ("exp", exp[nan_loc,nan_id :nan_id+5])
        # print("prevgrads", prevgrads)
        # print("generator weights: ", [x.data for x in generator.decoder.parameters()]) 
        # print("generator grad vals: ", [x.grad for x in generator.decoder.parameters()])
        sys.exit() 

    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

    scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
    discriminator_loss = g_loss_fn(scores_fake)

    if args.use_discriminator:
        print("using discriminator loss")
        loss += discriminator_loss
    losses['G_discriminator_loss'] = discriminator_loss.item()
    losses['G_total_loss'] = loss.item()

    optimizer_g.zero_grad()
    if curv_debug:
        print("generator grad vals BEFORE: ", [x.grad for x in generator.decoder.parameters()])
    loss.backward(retain_graph=True)

    for x in generator.decoder.parameters():
        nanmask = (x.grad != x.grad)
        # print(nanmask)
        x.grad.data[nanmask] = 0
        if torch.isnan(x.grad).any():
            print("NaN generated during backprop")
            print(x.grad)
            sys.exit()  

    if curv_debug:
        print("generator grad vals: ", [x.grad for x in generator.decoder.parameters()])
    if args.clipping_threshold_g > 0:
        nn.utils.clip_grad_norm_(
            generator.parameters(), args.clipping_threshold_g
        )
    
    optimizer_g.step()
    #print some grads here to see what's going wrong:
    if curv_debug:
        print("autograd.grad(loss, curvature_loss)", autograd.grad(loss, curvature_loss))
        # print("autograd.grad(loss, areas)", autograd.grad(loss, areas))
        # print("autograd.grad(loss, aside)", autograd.grad(loss, aside))
        # print("autograd.grad(curvature_loss, ccs)", autograd.grad(curvature_loss, ccs))
        print("autograd.grad(curvature_loss, pred_traj_fake_rel)", autograd.grad(curvature_loss, pred_traj_fake_rel))

    # sys.exit()
    return losses


def check_accuracy(
    args, loader, generator, discriminator, d_loss_fn, limit=False
):
    d_losses = []
    metrics = {}
    g_l2_losses_abs, g_l2_losses_rel = ([],) * 2
    disp_error, disp_error_l, disp_error_nl = ([],) * 3
    f_disp_error, f_disp_error_l, f_disp_error_nl = ([],) * 3
    total_traj, total_traj_l, total_traj_nl = 0, 0, 0
    loss_mask_sum = 0
    generator.eval()
    with torch.no_grad():
        for batch in loader:
            if args.use_gpu:
                batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch
            linear_ped = 1 - non_linear_ped
            loss_mask = loss_mask[:, args.obs_len:]

            pred_traj_fake_rel = generator(
                obs_traj, obs_traj_rel, seq_start_end
            )
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

            g_l2_loss_abs, g_l2_loss_rel = cal_l2_losses(
                pred_traj_gt, pred_traj_gt_rel, pred_traj_fake,
                pred_traj_fake_rel, loss_mask
            )
            ade, ade_l, ade_nl = cal_ade(
                pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
            )

            fde, fde_l, fde_nl = cal_fde(
                pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
            )

            traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
            traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
            traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
            traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

            scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
            scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)

            d_loss = d_loss_fn(scores_real, scores_fake)
            d_losses.append(d_loss.item())

            g_l2_losses_abs.append(g_l2_loss_abs.item())
            g_l2_losses_rel.append(g_l2_loss_rel.item())
            disp_error.append(ade.item())
            disp_error_l.append(ade_l.item())
            disp_error_nl.append(ade_nl.item())
            f_disp_error.append(fde.item())
            f_disp_error_l.append(fde_l.item())
            f_disp_error_nl.append(fde_nl.item())

            loss_mask_sum += torch.numel(loss_mask.data)
            total_traj += pred_traj_gt.size(1)
            total_traj_l += torch.sum(linear_ped).item()
            total_traj_nl += torch.sum(non_linear_ped).item()
            if limit and total_traj >= args.num_samples_check:
                break

    metrics['d_loss'] = sum(d_losses) / len(d_losses)
    metrics['g_l2_loss_abs'] = sum(g_l2_losses_abs) / loss_mask_sum
    metrics['g_l2_loss_rel'] = sum(g_l2_losses_rel) / loss_mask_sum

    metrics['ade'] = sum(disp_error) / (total_traj * args.pred_len)
    metrics['fde'] = sum(f_disp_error) / total_traj
    if total_traj_l != 0:
        metrics['ade_l'] = sum(disp_error_l) / (total_traj_l * args.pred_len)
        metrics['fde_l'] = sum(f_disp_error_l) / total_traj_l
    else:
        metrics['ade_l'] = 0
        metrics['fde_l'] = 0
    if total_traj_nl != 0:
        metrics['ade_nl'] = sum(disp_error_nl) / (
            total_traj_nl * args.pred_len)
        metrics['fde_nl'] = sum(f_disp_error_nl) / total_traj_nl
    else:
        metrics['ade_nl'] = 0
        metrics['fde_nl'] = 0

    generator.train()
    return metrics


def cal_l2_losses(
    pred_traj_gt, pred_traj_gt_rel, pred_traj_fake, pred_traj_fake_rel,
    loss_mask
):
    g_l2_loss_abs = l2_loss(
        pred_traj_fake, pred_traj_gt, loss_mask, mode='sum'
    )
    g_l2_loss_rel = l2_loss(
        pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode='sum'
    )
    return g_l2_loss_abs, g_l2_loss_rel


def cal_ade(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped):
    ade = displacement_error(pred_traj_fake, pred_traj_gt)
    ade_l = displacement_error(pred_traj_fake, pred_traj_gt, linear_ped)
    ade_nl = displacement_error(pred_traj_fake, pred_traj_gt, non_linear_ped)
    return ade, ade_l, ade_nl


def cal_fde(
    pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
):
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
    fde_l = final_displacement_error(
        pred_traj_fake[-1], pred_traj_gt[-1], linear_ped
    )
    fde_nl = final_displacement_error(
        pred_traj_fake[-1], pred_traj_gt[-1], non_linear_ped
    )
    return fde, fde_l, fde_nl


if __name__ == '__main__':
    print("POST AAMAS changes incoming..\n\n\n")
    
    train_log_dir = './logs/'+args.checkpoint_name +'_'+ current_time # [edit to suit experiment]
    writer = tf.summary.FileWriter(train_log_dir)
    main(args)
