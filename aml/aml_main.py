#
#
#
# A simple main for AML approach.
#
"""
python -m aml.aml_main --data_source Ellipse --save_path_prefix=/tmp/tmp_aml

python -m aml.aml_main --data_source Ellipse --max_syzygies 3

tensorboard --logdir=/tmp/tmp_aml --bind_all

"""
import numpy as np
import os
import torch
from torch.backends import cudnn

from datetime import datetime
from tensorboardX import SummaryWriter

from aml.aml_arguments import get_all_args
from aml.data_sources import (Ellipse,
                              ConservativeBlock1D,
                              ConservativeBlock2D,
                              ConservativeBlockConstVel1D,
                              ConservativeBlockConstVel2D,
                              ConservativeBlock45Incline1D,
                              Block36InclineNoDrag1D,
                              Block1D,
                              Block2D,
                              ConservativeBlockOnIncline,
                              BlockOnIncline,
                              BlockOnInclineODE,)

from aml.aml_core import AML, log_info


# import sys
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from mpl_toolkits.mplot3d import Axes3D
# import scipy.interpolate as interpolate
# import torch.distributed as dist
# import torch.multiprocessing as mp

np.set_printoptions(precision=4, linewidth=150, threshold=None, suppress=True)
torch.set_printoptions(precision=4, linewidth=150, threshold=500000)
data_sources = {'Ellipse': Ellipse,
                'ConservativeBlock1D': ConservativeBlock1D,
                'ConservativeBlock2D': ConservativeBlock2D,
                'ConservativeBlockConstVel1D': ConservativeBlockConstVel1D,
                'ConservativeBlockConstVel2D': ConservativeBlockConstVel2D,
                'ConservativeBlock45Incline1D': ConservativeBlock45Incline1D,
                'Block36InclineNoDrag1D': Block36InclineNoDrag1D,
                'Block1D': Block1D,
                'Block2D': Block2D,
                'ConservativeBlockOnIncline': ConservativeBlockOnIncline,
                'BlockOnIncline': BlockOnIncline,
                'BlockOnInclineODE': BlockOnInclineODE,
                }


def main():
    # Parse arguments
    args = get_all_args()
    # Set up torch
    torch_setup(args)
    # Set up logging.
    loggers = set_loggers(args)
    # Construct a data source.
    data_source_class = data_sources[args.data_source]
    # Start training.
    train(data_source_class, args, loggers)


def torch_setup(args):
    # Setup torch

    if not torch.cuda.is_available():
        args.device = 'cpu'

    if args.device != 'cpu':
        torch.cuda.set_device(args.device)
        torch.backends.cudnn.deterministic = False   # faster, less reproducible
        torch.cuda.manual_seed_all(args.seed)

    print(f'Initialized torch to use device {args.device}.')

    torch.manual_seed(args.seed)  # same seed for CUDA to get same model weights


def set_loggers(arguments):
    # Create file save path for this run.
    date_str = datetime.strftime(datetime.today(), "%y%m%d_%H%M%S")
    dir_parts = ['AML', arguments.data_source, date_str]
    arguments.save_path = os.path.join(
        os.path.expanduser(arguments.save_path_prefix), '_'.join(dir_parts))
    assert(not os.path.exists(arguments.save_path))
    os.makedirs(arguments.save_path)

    # Stdout and log file logging of arguments.
    log_file_name = os.path.join(arguments.save_path, 'log.txt')
    log_file = open(log_file_name, 'w', buffering=1)
    log_info(log_file, datetime.now().strftime('Start date %Y-%m-%d; arguments:'))
    log_info(log_file, arguments)

    # Tensor board logging.
    tensorboard_writer = SummaryWriter(arguments.save_path)
    arguments_string = ''
    for argument in vars(arguments):
        arguments_string += '  \n{:s}={:s}'.format(     # TB uses markdown-like
            str(argument), str(getattr(arguments, argument)))  # formatting, hence '  \n'
    tensorboard_writer.add_text('arguments', arguments_string, 0)

    return tensorboard_writer, log_file


def train(data_source_class, args, loggers):
    tensorboard_writer, logfile = loggers
    # Set up.
    batch_size = args.g_batch_size
    device = args.device
    data_source = data_source_class(batch_size, device,
                                    visualization_type=args.visualization_type,
                                    visualization_dim_inds=args.visualization_data_indexes)

    # Get test data.
    test_on_data = data_source.get_batch(batch_size=batch_size * 2,  noise_scale=0.0)
    test_off_data = data_source.get_batch(batch_size=batch_size * 2, noise_scale=0.0,
                                          on_manifold=False)
    test_data_info = [test_on_data, test_off_data]
    if hasattr(data_source, 'evaluate_a_g'):
        test_on_tgt = data_source.evaluate_a_g(test_on_data)
        test_off_tgt = data_source.evaluate_a_g(test_off_data)
        test_data_info.extend([test_on_tgt, test_off_tgt])

    # Start training.
    log_info(logfile, 'Training started...')
    aml = AML(len(data_source.dim_names), args.max_syzygies > 0)
    total_epochs = 0

    # Load checkpoint if needed.
    if args.load_checkpt is not None:
        epoch, total_epochs = aml.load(args.load_checkpt, args, device, logfile)
        if args.visualization_interval is not None:
            data_source.visualize(aml,  args.noise_scale,
                                  args.visualization_points,
                                  args.maximal_visualization_iterations,
                                  total_epochs,
                                  args.save_path, tensorboard_writer)

    # Train k relations.
    while aml.number_of_relations() < args.max_relations:
        # Add a (untrained) relation
        aml.add_g(args.g_hidden_size, args.g_num_layers,
                  args.g_lr, args.max_syzygies > 0, args.device, logfile)

        # Learn (aka train) the relation.
        g_max_epochs = args.g_max_train_epochs
        if aml.number_of_relations() > 1 and args.max_syzygies > 0:
            g_max_epochs = min(1000, args.g_max_train_epochs)
        total_epochs = learn_g(
            aml, data_source, args, 0, total_epochs, g_max_epochs,
            None, None, test_data_info, tensorboard_writer, logfile)

        if aml.number_of_relations() <= 1:
            continue
        # For a fixed number of times, learn a syzygy and re-learn the relation.
        for syz_i in range(args.max_syzygies):
            syz_holds, syz_noise, gjs_on_means = learn_syz(
                syz_i, aml, data_source, args, logfile)
            total_epochs = learn_g(
                aml, data_source, args, 0, total_epochs,
                args.g_max_train_epochs//args.max_syzygies,
                syz_noise, gjs_on_means, test_data_info,
                tensorboard_writer, logfile)
    log_info(logfile, 'Training done.')


def learn_g(aml, data_source, args, epoch, tot_epoch, g_max_train_epochs,
            syz_noise, gjs_on_means, test_data_info, tb_writer, logfile):

    bsz = args.g_batch_size
    dvc = args.device
    time_dict = {}
    init_on_sz = None
    on_sz = None

    while epoch < g_max_train_epochs:
        # Get training data.
        train_on_data = data_source.get_batch(bsz, dvc,
                                              noise_scale=args.noise_scale)
        train_off_data = data_source.get_batch(bsz, dvc,
                                               noise_scale=args.noise_scale,
                                               on_manifold=False)
        # Untrained outputs.
        on_out = aml.new_g_nn(train_on_data)
        off_out = aml.new_g_nn(train_off_data)

        # Initial losses.
        g_loss = None
        selfgrad_loss = None
        on_loss = None

        on_grad = None
        is_last = (epoch+1 == g_max_train_epochs)
        dbg_dict = {} if (tot_epoch % args.log_interval == 0 or is_last) else None

        # Set up the loss, depending on mode/version.
        #
        # Pre-training
        if epoch < args.g_pretrain_epochs:
            assert False  # not using now
            on_tgt = data_source.evaluate_a_g(train_on_data)
            off_tgt = data_source.evaluate_a_g(train_off_data)
            g_loss = (aml.g_pretrain_loss(on_tgt, on_out) +
                      aml.g_pretrain_loss(off_tgt, off_out))

        else:
            selfgrad_loss, on_loss, done, on_grad = aml.g_on_loss(
                train_on_data, on_out, off_out)
            g_loss = selfgrad_loss + on_loss

            if not args.ablation:
                # Transversality.
                if args.max_syzygies <= 0:
                    g_loss, done, dbg_dict = aml.g_loss_with_transverse(
                        train_on_data, on_out, off_out, args.transverse_beta,
                        dbg_dict)

                # Restricted syzygies.
                else:
                    if done and dbg_dict is None:
                        dbg_dict = {}
                    if aml.number_of_relations() > 1 and \
                            aml.number_of_syzygies() > 0:
                        # augment g_loss by syz_move_g_loss to push from syz
                        _, syz_move_g_loss, syz_holds, on_sz, dbg_dict = \
                            aml.syzygy_loss(train_on_data, syz_noise,
                                            gjs_on_means, False, on_loss,
                                            args.syz_move_g_beta, dbg_dict,
                                            time_dict)
                        g_loss = g_loss + syz_move_g_loss
                        if init_on_sz is None:
                            init_on_sz = on_sz
                        if np.abs(on_sz-init_on_sz) < 0.5*init_on_sz:
                            done = False  # no shortcut if g has not moved much
                        if on_sz > 2*init_on_sz:
                            done = True  # don't push too far

        # Store some data for debugging and visualizing.
        # Make save_debug_dict(done, dbg_dict)
        if done and dbg_dict is None:
            dbg_dict = {}
        if dbg_dict is not None:
            dbg_dict['g_loss'] = g_loss.mean()
            if selfgrad_loss is not None:
                dbg_dict['selfgrad_loss'] = selfgrad_loss
                dbg_dict['on_loss'] = on_loss
                dbg_dict['on_grad'] = torch.abs(on_grad).mean()
                dbg_dict['on_grad_norm'] = on_grad.norm(dim=1).mean()
            if init_on_sz is not None:
                dbg_dict['init_on_sz'] = init_on_sz
                dbg_dict['on_sz'] = on_sz
        # Uncomment code below to print gradients for manual inspection.
        # if tot_epoch%arguments.log_interval==0 and aml.k()>1:
        #    # print grads *before* optim step, so right after loss compute
        #    print_grads(aml.new_g_nn, aml.new_g_optim, dbg_dict)

        # Backprop (for the most basic pytorch example with correct order see:
        # https://pytorch.org/tutorials/beginner/pytorch_with_examples.html )
        aml.new_g_optim.zero_grad()
        g_loss.backward()
        aml.new_g_optim.step()

        # Logging, saving.
        if epoch+1 == args.g_pretrain_epochs:
            log_info(logfile, 'pretrain done')
        if done:
            msg = 'learn_g done mean on {:0.4f} off {:0.4f}'
            log_info(logfile, msg.format(aml.new_g_nn.on_mean[0],
                                         aml.new_g_nn.off_mean[0]))
        if tot_epoch % args.log_interval == 0 or is_last:
            msg = 'g_{:d} epoch {:d} tot_epoch {:d} g_loss {:0.9f}'
            log_info(logfile, msg.format(aml.number_of_relations() - 1,
                                         epoch, tot_epoch, g_loss))
            on_out_bottom = torch.abs(on_out).squeeze()
            if bsz > 10:
                on_out_bottom, _ = on_out_bottom.topk(k=bsz//5)
            dbg_dict['on_out_bottom_mean'] = on_out_bottom.mean()
            dbg_dict['on_out_mean'] = torch.abs(on_out).mean()
            dbg_dict['off_out_mean'] = torch.abs(off_out).mean()
            log_str = ''
            for k, v in dbg_dict.items():
                if v is None:
                    v = 0.0
                if v is not None:
                    log_str += '{:s} {:0.4f} '.format(k, v)
                tb_writer.add_scalar(k, v, tot_epoch)
            log_info(logfile, log_str)
            if args.max_syzygies > 0 and len(time_dict) > 0:
                print('syz time: ', end='')
                for k, v in time_dict.items():
                    print('{:s} {:0.4f}'.format(k, v), end=' ')
                print('tot sec\n', end='')
        if (args.visualization_interval is not None and
                (epoch > 0 or aml.number_of_relations() == 1) and
                (tot_epoch % args.visualization_interval == 0 or is_last or done)):
            print_test_info(aml, tot_epoch, test_data_info, logfile)
            data_source.visualize(aml, args.noise_scale,
                                  args.visualization_points,
                                  args.maximal_visualization_iterations,
                                  tot_epoch,
                                  args.save_path, tb_writer)
            aml.save(args, epoch, tot_epoch, logfile)
        epoch += 1
        tot_epoch += 1

        if done:
            break  # done training this g relation

    # Save last even if we don't visualize or log.
    if args.visualization_interval is None:
        aml.save(args, epoch, tot_epoch, logfile)
        msg = 'learn_g last mean on {:0.4f} off {:0.4f}'
        log_info(logfile,
                 msg.format(aml.new_g_nn.on_mean[0], aml.new_g_nn.off_mean[0]))

    return tot_epoch


def learn_syz(syz_i, aml, data_source, args, logfile):
    # Init parameters
    bsz = args.g_batch_size
    dvc = args.device
    syz_max_train_epochs = args.syz_max_train_epochs
    if syz_max_train_epochs is None:
        syz_max_train_epochs = args.g_max_train_epochs//2

    # Add an untrained syzygy.
    aml.add_syzygy(syz_i, len(data_source.dim_names),
                   args.max_relations, args.syz_hidden_size,
                   args.g_num_layers, args.g_lr, args.device, logfile)

    on_data = data_source.get_batch(bsz, dvc, noise_scale=args.noise_scale)
    syz_noise, gjs_on_means = aml.syzygy_thickening_info(on_data, logfile)

    # Train the syzygy.
    for epoch in range(syz_max_train_epochs):
        dbg_dict = {} if (epoch % args.log_interval) == 0 else None
        train_on_data = data_source.get_batch(bsz, dvc,
                                              noise_scale=args.noise_scale)
        syz_loss, _, syz_holds, _, dbg_dict = aml.syzygy_loss(
            train_on_data, syz_noise, gjs_on_means, True, None, 0, dbg_dict)
        syz_loss = args.syz_beta*syz_loss
        aml.syz_optim.zero_grad()
        syz_loss.backward()
        aml.syz_optim.step()
        log_str = 'syz train epoch {:d} '.format(epoch)
        if dbg_dict is not None:
            for k, v in dbg_dict.items():
                if v is not None:
                    log_str += '{:s} {:0.4f} '.format(k, v)
            log_info(logfile, log_str)
        if epoch > syz_max_train_epochs//10 and syz_holds[-1]:
            break
    return syz_holds[-1], syz_noise, gjs_on_means


def print_test_info(aml, epoch, test_data_info, logfile):
    nprint = 5
    has_lbls = (len(test_data_info) >= 4)
    test_on_data, test_off_data = test_data_info[0], test_data_info[1]
    if has_lbls:
        test_on_tgt = test_data_info[2]
        test_off_tgt = test_data_info[3]
    log_info(logfile, 'epoch {:d}'.format(epoch))
    test_on_out = aml.new_g_nn(test_on_data)
    sz = min(nprint, test_on_data.size(0))
    log_info(logfile, 'test_on_data')
    log_info(logfile, test_on_data[0:sz].squeeze().detach().cpu().numpy())
    if has_lbls:
        log_info(logfile, 'test_on_lbl')
        log_info(logfile, test_on_tgt[0:sz].squeeze().detach().cpu().numpy())
    log_info(logfile, 'test_on_out')
    log_info(logfile, test_on_out[0:sz].squeeze().detach().cpu().numpy())
    test_off_out = aml.new_g_nn(test_off_data)
    sz = min(nprint, test_off_data.size(0))
    log_info(logfile, 'test_off_data')
    log_info(logfile, test_off_data[0:sz].squeeze().detach().cpu().numpy())
    if has_lbls:
        log_info(logfile, 'test_off_lbl')
        log_info(logfile, test_off_tgt[0:sz].squeeze().detach().cpu().numpy())
    log_info(logfile, 'test_off_out')
    log_info(logfile, test_off_out[0:sz].squeeze().detach().cpu().numpy())
    if has_lbls:
        test_on_loss = aml.g_pretrain_loss(test_on_tgt, test_on_out)
        test_off_loss = aml.g_pretrain_loss(test_off_tgt, test_off_out)
        msg = 'test_on_loss {:0.4f} test_off_loss {:0.4f}'
        log_info(logfile, msg.format(test_on_loss, test_off_loss))


if __name__ == '__main__':
    main()
