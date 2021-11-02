#
#
# AML core code.
#

import os
# import sys
# import time
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from mpl_toolkits.mplot3d import Axes3D
# import scipy.interpolate as interpolate
import torch
# import torch.distributed as dist
# import torch.multiprocessing as mp
from datetime import datetime

from aml.aml_nets import GNN
from aml.aml_nets import RestrictedSyzygy

np.set_printoptions(precision=4, linewidth=150, threshold=None, suppress=True)
torch.set_printoptions(precision=4, linewidth=150, threshold=500000)


def log_info(logfile, data):
    """Set up for logging to file and tensorboard."""
    tm = datetime.now().strftime('%H:%M:%S')
    print(tm, data)
    logfile.write(tm + ' ' + str(data) + '\n')


def optim_to_cuda(optim, device):
    for st in optim.state.values():
        for k, v in st.items():
            if torch.is_tensor(v):
                st[k] = v.to(device)


class AML:
    """Trains the relations and the syzygies."""

    def __init__(self, g_inp_ndims, do_syz=False):
        self.g_inp_ndims = g_inp_ndims
        self.new_g_nn = None  # NN for new g_k
        self.new_g_optim = None  # optimizer for new g_k
        self.gj_nns = []  # previously learned relations g_1,...,g_{k-1}
        self.gj_optims = []  # optimizers of previously learned relations

        self.syz_nn = None  # NN for new syzygy
        self.syz_optim = None  # optimizer for new syzygy
        self.syzj_nns = []  # previously learned relations g_1,...,g_{k-1}
        self.syzj_optims = []  # optimizers of previously learned relations

        self._g_onvalue_beta = 1e-3  # multipliers to bring on-manifold values
        self._g_selfgrad_beta = 1e2  # and gradient norms in the same range
        self._g_selfgrad_margin = 1e-4  # if do_syz else 1e-2  # grad norm target
        self._g_transverse_margin = 0.4 * (np.pi / 2.0)  # min target angle
        self._eps = 1e-15  # eps for numerical stability (needs to be small)
        self._out_f_max = 100.0  # max for syzygy pre-outputs and transverse output

    def get_on_off_means(self):
        """"On-manifold and off-manifold means of the k relations."""
        on_means, off_means, k = [], [], self.number_of_relations()
        for j in range(k):
            g_nn = self.new_g_nn if (j + 1) == k else self.gj_nns[j]
            on_means.append(g_nn.on_mean)
            off_means.append(g_nn.off_mean)
        on_means = torch.cat(on_means, dim=0)
        off_means = torch.cat(off_means, dim=0)
        return on_means, off_means

    def add_g(self, g_hidden_size, g_num_layers, g_lr, do_syz, device, logfile):
        # 2**6 = 64, g_hidden_size default is 4
        gh = [g_hidden_size * (2 ** min(self.number_of_relations(), 6))] * g_num_layers
        if self.new_g_nn is not None:
            self.gj_nns.append(self.new_g_nn)
            self.gj_optims.append(self.new_g_optim)
        if self.syz_nn is not None:  # clean up syzygies learned for previous g
            self.syz_nn = None
            self.syz_optim = None
            self.syzj_nns.clear()
            self.syzj_optims.clear()
        activation = torch.nn.Tanh()  # if do_syz else torch.nn.ELU()
        out_nl = torch.nn.Tanh()  # if do_syz else None

        # Note: ELU causes high on-manifold values even without noise,
        # but Sigmoid works well, with on-manifold values ~0 without noise.
        self.new_g_nn = GNN(self.g_inp_ndims, 1, hl_output_sizes=gh, activation=activation, out_activation=out_nl, device=device)
        self.new_g_optim = torch.optim.Adam(
            [{'params': self.new_g_nn.parameters(), 'lr': g_lr}])

        # Log the construction.
        msg = '++++++++++++ Constructed g_{:d} ++++++++++++\n'
        msg += str(self.new_g_nn)
        log_info(logfile, msg.format(self.number_of_relations() - 1))

        return self.new_g_nn, self.new_g_optim

    def add_syzygy(self, syzi, ndims, max_relations, syz_hidden_size,
                   g_num_layers, g_lr, device, logfile):
        assert (self.number_of_relations() > 1)  # make sure we finished learning 1st relation
        if self.syz_nn is not None:
            self.syzj_nns.append(self.syz_nn)
            self.syzj_optims.append(self.syz_optim)
        syzh = [syz_hidden_size] * g_num_layers
        # print(f'syzh={syzh}')
        self.syz_nn = RestrictedSyzygy(
            in_size=ndims, hl_output_sizes=syzh, nl=torch.nn.ELU(),
            k=max_relations, f_max=self._out_f_max, device=device)
        self.syz_optim = torch.optim.Adam(
            [{"params": self.syz_nn.parameters(), "lr": g_lr}])

        # Log the construction.
        msg = '++++++++++++ Constructed syz_{:d} ++++++++++++'
        msg += str(self.syz_nn)
        log_info(logfile, msg.format(self.number_of_syzygies() - 1))

        return self.syz_nn, self.syz_optim

    def g_pretrain_loss(self, tgt, out):
        """L1 distance between output and target."""
        assert (tgt.size() == out.size())
        loss = torch.abs(tgt - out)  # L1 to distinguish on- vs off-manifold
        return loss.mean()

    def g_on_loss(self, on_data, on_out, off_out):
        # Compute v_k vector for current g_k, *do not detach*
        self.new_g_optim.zero_grad()
        on_data_with_grad = on_data.clone().detach().requires_grad_(True)
        g_out = self.new_g_nn(on_data_with_grad)
        on_grad = torch.autograd.grad(
            g_out.mean(), on_data_with_grad, create_graph=True)[0]
        on_grad_norm = on_grad.norm(dim=1)

        # Compute selfgrad_loss.
        grad_for_loss = torch.clamp(on_grad_norm, 0, self._g_selfgrad_margin)
        selfgrad_loss = 1.0 - self._g_selfgrad_beta * grad_for_loss

        # Compute on_loss.
        on_grad_norm = on_grad.norm(dim=1, keepdim=True)
        on_grad_norm = torch.clamp(on_grad_norm, self._eps, float('Inf'))
        on_loss = self._g_onvalue_beta * torch.abs(on_out) / on_grad_norm
        on_loss = torch.clamp(on_loss, 0.05, float('Inf'))  # hinge loss
        done = self.selfgrad_at_margin(on_grad) and \
            self.on_off_done(on_out, off_out)
        on_abs_mean = torch.abs(on_out).mean().item()
        off_abs_mean = torch.abs(off_out).mean().item()
        alph = 0.1  # moving average decay factor (can by anything in [0,1])
        self.new_g_nn.on_mean[0] = (1 - alph) * self.new_g_nn.on_mean[0] + alph * on_abs_mean
        self.new_g_nn.off_mean[0] = (1 - alph) * self.new_g_nn.off_mean[0] + alph * off_abs_mean

        return selfgrad_loss.mean(), on_loss.mean(), done, on_grad

    def selfgrad_at_margin(self, on_grad):
        return on_grad.norm(dim=1).mean() >= 0.9 * self._g_selfgrad_margin

    def on_off_done(self, on_out, off_out):
        bsz = on_out.size(0)
        num_bottom = bsz // 5 if bsz > 10 else bsz
        on_abs_bottom, _ = torch.abs(on_out).squeeze().topk(k=num_bottom)
        separated = 5 * on_abs_bottom.mean() < torch.abs(off_out).mean()
        return separated

    def g_loss_with_transverse(self, on_data, on_out, off_out, beta, dbg_dict):
        selfgrad_loss, on_loss, done, on_grad = self.g_on_loss(
            on_data, on_out, off_out)
        total_loss = selfgrad_loss + on_loss
        # Compute transverse_loss if needed.
        if len(self.gj_nns) > 0:
            gj_grads, _ = get_gj_grads(on_data, self.gj_nns, self.gj_optims)
            transverse_loss_value, mean_angle = transverse_loss(on_data, on_grad, gj_grads, self._eps, dbg_dict)
            # Make sure that transverse_loss does not overpower on_loss.
            if beta is None:
                max_abs_loss = self._out_f_max * torch.abs(on_loss.detach())
                curr_transverse_loss = torch.clamp(
                    max_abs_loss - transverse_loss_value, 0, float('Inf'))
                curr_transverse_loss = (1 / self._out_f_max) * curr_transverse_loss
            else:
                curr_transverse_loss = beta * transverse_loss_value
            total_loss = selfgrad_loss + on_loss + curr_transverse_loss
            done = done and (mean_angle >= self._g_transverse_margin)
        if done and dbg_dict is None:
            dbg_dict = {}
        if dbg_dict is not None:
            dbg_dict['selfgrad_loss'] = selfgrad_loss
            dbg_dict['on_loss'] = on_loss
            dbg_dict['on_grad'] = on_grad.norm(dim=1).mean()
        return total_loss.mean(), done, dbg_dict

    def syzygy_thickening_info(self, on_data, logfile):
        # for k in ['syz_intro', 'syz_noise', 'syz_nn']:
        #    if k not in time_dict: time_dict[k] = 0
        # strt = time.time()
        # First find mean activation value for gs on-manifold
        # (normalized by the gradient at the value)
        gj_grads, gj_outs = get_gj_grads(on_data, self.gj_nns, self.gj_optims)
        gjs_on_means = torch.abs(torch.cat(gj_outs, dim=1)).mean(dim=0)
        gj_outs_normed = []
        for j in range(len(self.gj_nns)):
            gj_grad_norm = torch.clamp(
                gj_grads[j].norm(dim=1, keepdim=True), self._eps, float('Inf'))
            gj_outs_normed.append(gj_outs[j] / gj_grad_norm)
        gjs_on_mean_normed = torch.abs(torch.cat(gj_outs_normed, dim=1)).mean()
        # Find the noise level that will make gj activations large enough.
        mins, maxs = on_data.min(axis=0)[0], on_data.max(axis=0)[0]
        # time_dict['syz_intro'] += time.time()-strt; strt = time.time()
        wide_data, gj_wide_outs = None, None
        best_noise = 0.1
        for noise in np.linspace(0.1, 2.1, 10):
            wide_data = on_data + (torch.rand_like(on_data) - 0.5) * (maxs - mins) * noise
            gj_wide_grads, gj_wide_outs = get_gj_grads(
                wide_data, self.gj_nns, self.gj_optims)
            gjs_wide_mean = torch.abs(torch.cat(gj_wide_outs, dim=1)).mean()
            gj_wide_outs_normed = []
            for j in range(len(self.gj_nns)):
                gj_wide_grad_norm = torch.clamp(
                    gj_wide_grads[j].norm(dim=1, keepdim=True),
                    self._eps, float('Inf'))
                gj_wide_outs_normed.append(gj_wide_outs[j] / gj_wide_grad_norm)
            gjs_wide_mean_normed = torch.abs(
                torch.cat(gj_wide_outs_normed, dim=1)).mean()
            if gjs_wide_mean_normed > 2.0 * gjs_on_mean_normed:
                best_noise = noise
                break  # wide enough
        # time_dict['syz_noise'] += time.time()-strt; strt = time.time()
        log_info(logfile, f'syz_noise {best_noise:0.4f}')
        print(f'gjs_on_means {gjs_on_means.detach().cpu().numpy()}')
        return best_noise, gjs_on_means.detach()

    def syzygy_loss(self, on_data, noise, gjs_on_means, syz_train,
                    on_loss, syz_move_g_beta, dbg_dict, time_dict=None):
        mins, maxs = on_data.min(axis=0)[0], on_data.max(axis=0)[0]
        wide_data = on_data + (torch.rand_like(on_data) - 0.5) * (maxs - mins) * noise
        gj_wide_outs = []
        for gj_nn in self.gj_nns:
            gj_wide_outs.append(gj_nn(wide_data))
        gj_wide_outs = torch.cat(gj_wide_outs, dim=1)

        # Forward pass on syzygies.
        g_wide_out = self.new_g_nn(wide_data)
        syz_loss = None
        syz_sin_mean = None
        syz_move_g_losses = []
        syz_mean = None
        syz_holds = []
        f_outs = []
        on_sz = None
        n = self.number_of_syzygies()
        # pairwise_syz_mean = None
        for j in range(n):
            syz_nn = self.syz_nn if (j + 1) == n else self.syzj_nns[j]
            syz_out, f_out, on_sz = syz_nn(
                wide_data, gj_wide_outs, gjs_on_means, g_wide_out, dbg_dict)
            syz_mean = torch.abs(syz_out).mean()
            syz_combo_abs = syz_mean
            # if len(pairwise_syz_mean)>0:
            #    syz_combo_abs = syz_mean + sum(pairwise_syz_mean)
            f_outs.append(f_out)
            where_holds = (syz_mean < gjs_on_means.mean())
            syz_holds.append(where_holds.item())
            if syz_train:
                continue
            # Make sure that move_g_loss does not overpower on_loss.
            max_abs_loss = self._out_f_max * syz_move_g_beta * torch.abs(on_loss.detach())
            curr_move_g_loss = torch.clamp(max_abs_loss - syz_combo_abs, 0, float('Inf'))
            curr_move_g_loss = (1 / self._out_f_max) * curr_move_g_loss
            # if dbg_dict is not None:
            #    print('max_abs_loss', max_abs_loss.mean())
            #    print('on_loss', torch.abs(on_loss.detach()))
            #    print('max_abs_loss-syz_mean', max_abs_loss-syz_mean)
            #    print('curr_move_g_loss', curr_move_g_loss)
            syz_move_g_losses.append(curr_move_g_loss)
        if syz_train:
            syz_loss = syz_mean
            # if len(pairwise_syz_mean)>0:
            #    syz_loss = syz_mean + sum(pairwise_syz_mean)
        # Putting all losses in a list and summing should work:
        # https://discuss.pytorch.org/t/
        # how-to-combine-multiple-criterions-to-a-loss-function/348
        syz_move_g_loss = None
        if len(syz_move_g_losses) > 0:
            syz_move_g_loss = sum(syz_move_g_losses)
        # time_dict['syz_nn'] += time.time()-strt; strt = time.time()
        if dbg_dict is not None:
            dbg_dict['gjs_on_mean'] = gjs_on_means.mean()
            dbg_dict['syz_noise'] = noise
            dbg_dict['syz_mean'] = syz_mean
            dbg_dict['on_sz'] = on_sz
            # if len(pairwise_syz_mean)>0:
            #    dbg_dict['pairwise_syz_mean_sum'] = sum(pairwise_syz_mean)
            for syzi, syzhld in enumerate(syz_holds):
                dbg_dict['syz' + str(syzi) + '_holds'] = syz_holds[syzi]
            if syz_loss is not None:
                dbg_dict['syz_loss'] = syz_loss
            if syz_sin_mean is not None:
                dbg_dict['syz_sin_mean'] = syz_sin_mean
            if syz_move_g_loss is not None:
                dbg_dict['syz_move_g_loss'] = syz_move_g_loss
        return syz_loss, syz_move_g_loss, syz_holds, on_sz, dbg_dict

    def latent_transfer_loss(self, z_smpls, act_1toT, static_sz,
                             aml_static_sz, dbg_dict, dbg_pfx=''):
        # Hard-code means for now, but need to load later.
        dvc = z_smpls.device
        bsz, seq_len, ndim = z_smpls.size()
        # For now deal with T=2, but this can be easily generalized.
        assert (seq_len == 2)
        f_smpl = z_smpls[:, 0, :static_sz]
        act = act_1toT[:, 0, :]
        # For static states: we could make sure they agree.
        # static_loss = torch.abs(f_smpl-z_smpls[:,1,:static_sz]).mean()
        # Pass one static (for incline theta) and all dynamic latents
        # through g relations from AML. The loss is how far
        # from 0 the output is, since gs should hold for good latent repr.
        g_inp = torch.cat(
            [z_smpls[:, 0, static_sz:], z_smpls[:, 1, static_sz:], act], dim=1)
        if aml_static_sz > 0:
            st = f_smpl[:, static_sz - aml_static_sz:static_sz]
            g_inp = torch.cat([st, g_inp], dim=1)
        aml_outs = []
        k = self.number_of_relations()
        on_means, off_means = self.get_on_off_means()
        for j in range(k):
            g_nn = self.new_g_nn if (j + 1) == k else self.gj_nns[j]
            aml_outs.append(g_nn(g_inp))
        aml_out_abs = torch.abs(torch.cat(aml_outs, dim=1))
        dynamic_loss = torch.where(
            aml_out_abs > on_means, aml_out_abs, torch.zeros_like(aml_out_abs))
        dynamic_loss = dynamic_loss.mean(dim=1)
        aml_loss = dynamic_loss  # + static loss
        aml_ok = torch.where(aml_out_abs < on_means,
                             torch.ones_like(aml_out_abs),
                             torch.zeros_like(aml_out_abs))
        aml_ok = aml_ok.min(dim=1)[0]
        ok_ids = torch.nonzero(aml_ok, as_tuple=True)[0]
        aml_almost_ok = torch.where(aml_out_abs < 2 * on_means,
                                    torch.ones_like(aml_out_abs),
                                    torch.zeros_like(aml_out_abs))
        aml_almost_ok = aml_almost_ok.min(dim=1)[0]
        almost_ok_ids = torch.nonzero(aml_almost_ok, as_tuple=True)[0]
        # if dbg_dict is not None:
        #    print('ok_ids', len(ok_ids), 'output of', bsz)
        #    print('aml_out_abs\n', aml_out_abs[ok_ids[0:5]])
        #    print('dynamic_loss\n', dynamic_loss[ok_ids[0:5]])
        if dbg_dict is not None:
            # dbg_dict[dbg_pfx+'aml_static_loss'] = static_loss
            dbg_dict[dbg_pfx + 'aml_dynamic_loss'] = dynamic_loss.mean()
            dbg_dict[dbg_pfx + 'aml_ok_frac'] = len(ok_ids) / bsz
            dbg_dict[dbg_pfx + 'aml_almost_ok_frac'] = len(almost_ok_ids) / bsz
        return aml_loss, ok_ids

    def number_of_relations(self):
        return int(self.new_g_nn is not None) + len(self.gj_nns)

    def number_of_syzygies(self):
        return int(self.syz_nn is not None) + len(self.syzj_nns)

    def save(self, args, epoch, tot_epoch, logfile, checkpt_path=None):
        if self.new_g_nn is None:
            return
        save_dict = {'g_nn_state_dict': self.new_g_nn.state_dict(),
                     'g_optim_state_dict': self.new_g_optim.state_dict(),
                     'epoch': epoch, 'tot_epoch': tot_epoch, 'arguments': args}
        for j, gj_nn, gj_optim in zip(
                list(range(len(self.gj_nns))), self.gj_nns, self.gj_optims):
            save_dict['gj' + str(j) + '_nn_state_dict'] = gj_nn.state_dict()
            save_dict['gj' + str(j) + '_optim_state_dict'] = gj_optim.state_dict()
        if self.syz_nn is not None:
            save_dict['syz_nn_state_dict'] = self.syz_nn.state_dict()
            save_dict['syz_optim_state_dict'] = self.syz_optim.state_dict()
            for j, syzj_nn, syzj_optim in zip(list(range(len(self.syzj_nns))),
                                              self.syzj_nns, self.syzj_optims):
                pfx = 'syzj' + str(j)
                save_dict[pfx + '_nn_state_dict'] = syzj_nn.state_dict()
                save_dict[pfx + '_optim_state_dict'] = syzj_optim.state_dict()

        if not checkpt_path:
            checkpt_path = args.save_path
        sfx = 'aml'
        sfx += '-g{:d}-ep{:d}.pt'.format(self.number_of_relations() - 1, epoch)
        checkpt_path = os.path.join(checkpt_path, sfx)
        log_info(logfile, f'AML saving checkpt in {checkpt_path}...')
        torch.save(save_dict, checkpt_path)
        log_info(logfile, 'AML saving done')

    def load(self, checkpt_path, args, device, logfile):
        checkpt_path = os.path.expanduser(checkpt_path)
        log_info(logfile, f"Loading chkpt {checkpt_path}...")
        assert (os.path.isfile(checkpt_path))
        chkpt = torch.load(checkpt_path, map_location=device)
        epoch = chkpt['epoch']
        tot_epoch = chkpt['tot_epoch']
        loaded_args = chkpt['arguments']
        if args is None:
            args = loaded_args
        for j in range(args.max_relations + 1):
            pfx = 'gj' + str(j) if j < args.max_relations else 'g'
            k = pfx + '_nn_state_dict'
            if k not in chkpt:
                continue
            self.add_g(args.g_hidden_size, args.g_num_layers,
                       args.g_lr, args.max_syzygies > 0, device, logfile)
            self.new_g_nn.load_state_dict(chkpt[k])
            self.new_g_optim.load_state_dict(chkpt[k.replace('_nn_', '_optim_')])
            if device != 'cpu':
                optim_to_cuda(self.new_g_optim, device)
        for j in range(args.max_syzygies + 1):
            pfx = 'syzj' + str(j) if j < args.max_syzygies else 'syz'
            k = pfx + '_nn_state_dict'
            if k not in chkpt:
                continue
            self.add_syzygy(j, self.g_inp_ndims, args.max_relations,
                            args.syz_hidden_size, args.g_num_layers,
                            args.g_lr, device, logfile)
            self.syz_nn.load_state_dict(chkpt[k])
            self.syz_optim.load_state_dict(chkpt[k.replace('_nn_', '_optim_')])
            if device != 'cpu':
                optim_to_cuda(self.syz_optim, device)

        # Log the load.
        on_means, off_means = self.get_on_off_means()
        msg = 'AML loaded {:d} relations, {:d} syzigies, epoch {:d}'
        msg += ' tot_epoch {:d} on/off means:'
        log_info(logfile, msg.format(self.number_of_relations(), self.number_of_syzygies(), epoch, tot_epoch))
        log_info(logfile, on_means.detach().cpu().numpy())
        log_info(logfile, off_means.detach().cpu().numpy())

        return epoch, tot_epoch


def get_gj_grads(on_data, gj_nns, gj_optims):
    gj_grads, gj_outs = [], []
    for gj_nn, gj_optim in zip(gj_nns, gj_optims):
        gj_optim.zero_grad()
        on_data_with_grad = on_data.clone().detach().requires_grad_(True)
        gj_out = gj_nn(on_data_with_grad)
        gj_grad = torch.autograd.grad(
            gj_out.mean(), on_data_with_grad, create_graph=True)[0]
        gj_grads.append(gj_grad.detach())
        gj_outs.append(gj_out.detach())
    return gj_grads, gj_outs


def transverse_loss(on_data, on_grad, gj_grads, eps, dbg_dict):
    assert (len(gj_grads) > 0)
    # gj_grads contain gradient of each g_j wrt train_on_data and detach
    # note: make sure on_grad had require_gradient=True
    v_k = on_grad
    # Compute dot products of v_k with each of gj_grads.
    cos_fn = torch.nn.CosineSimilarity(dim=1, eps=eps)
    log_sinsq_lst = []
    angle_lst = []
    for v_j in gj_grads:
        # https://discuss.pytorch.org/t/dot-product-batch-wise/9746/11
        cos = cos_fn(v_j, v_k)
        if not torch.isfinite(cos).all():
            print(cos)
        angle = torch.acos(torch.abs(cos)).mean().detach()  # for stopping
        angle_lst.append(angle)
        sin_sq = 1 - cos ** 2  # theta_{j,k} ~ sin^2 = 1-cos^2
        sin_sq = torch.where(sin_sq > 0, sin_sq, eps * torch.ones_like(sin_sq))
        assert ((sin_sq > 0.0).all() and (sin_sq <= 1.1).all())  # sanity checks
        log_sinsq_lst.append(torch.log(sin_sq))
        if dbg_dict is not None:  # print cos and angles
            np.set_printoptions(precision=9)
            for tmpi in range(2):
                tmp_angles = torch.acos(torch.abs(cos))
                print('cos {:0.9f} angle {:0.9f}\n'.format(
                    cos[tmpi], tmp_angles[tmpi]))
                print('v_j\n', v_j[tmpi].detach().cpu().numpy())
                print('v_k\n', v_k[tmpi].detach().cpu().numpy())
            np.set_printoptions(precision=4)
    mean_angle = torch.stack(angle_lst).mean()
    # To construct transversality loss:
    # log_sinsq_lst contains logs of quantities that we want to multiply:
    # selfgrad * sin(theta_{1,k})^2 * ... * sin(theta_{k-1,k})^2
    # The product of squared sines multiplied by squared norm of v_k
    # is approximately the area of the paralleloegram formed by all
    # the relations. We need to maximize this area, so subtract from loss.
    # Putting all losses in a list and summing should work:
    # https://discuss.pytorch.org/t/
    # how-to-combine-multiple-criterions-to-a-loss-function/348
    sin_prod = torch.sqrt(torch.exp(sum(log_sinsq_lst)))
    transverse_loss_value = 1.0 - sin_prod
    if dbg_dict is not None:
        dbg_dict['angle'] = mean_angle
        dbg_dict['sin_prod'] = sin_prod.mean()
        dbg_dict['transverse_loss_value'] = transverse_loss_value.mean()

    return transverse_loss_value.mean(), mean_angle
