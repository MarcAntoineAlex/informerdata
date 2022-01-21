""" Architect controls architecture of cell by computing gradients of alphas """
import copy

import torch
import torch.nn as nn
import torch.distributed as dist
import utils.tools as tools
from models.model import sigtemp
from torch.nn.functional import sigmoid


class Architect_Scinet():
    """ Compute gradients of alphas """
    def __init__(self, net, device, args, criterion, inverse_transform=None):
        """
        Args:
            net
            w_momentum: weights momentum
        """
        self.device = device
        self.args = copy.deepcopy(args)
        self.criterion = criterion
        if type(net) == nn.parallel.DistributedDataParallel:
            self.net_in = net
            self.net = self.net_in.module
            self.v_net = copy.deepcopy(net)
        else:
            self.net = net.to(self.device)
            self.v_net = copy.deepcopy(net)
        self.w_momentum =self.args.w_momentum
        self.w_weight_decay = self.args.w_weight_decay
        if self.args.inverse:
            self.inverse_transform = inverse_transform

    def critere(self, pred, true, indice, reduction='mean'):
        if self.args.fourrier:
            weights = self.net.arch()[indice, :, :]
            weights = sigtemp(weights, self.args.temp) * self.args.sigmoid
        else:
            weights = self.net.arch[indice, :, :]
            # weights = self.net.normal_prob(self.net.arch)[indice, :, :]
            weights = sigmoid(weights) * self.args.sigmoid
        if reduction != 'mean':
            crit = nn.MSELoss(reduction=reduction)
            return (crit(pred, true) * weights).mean(dim=(-1, -2))
        else:
            crit = nn.MSELoss(reduction='none')
            return (crit(pred, true) * weights).mean()

    def virtual_step(self, trn_data, trn_set, next_data, next_set, xi, w_optim, indice):
        """
        Compute unrolled weight w' (virtual step)

        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient

        Args:
            xi: learning rate for virtual gradient step (same as weights lr)
            w_optim: weights optimizer
        """
        # forward & calc loss
        pred = torch.zeros(trn_data[1][:, -self.args.pred_len:, :].shape).to(self.device)
        if self.args.rank == 0:
            pred, _, _, _, true, _ = self._process_one_batch_SCINet(trn_set, trn_data, self.net)
            unreduced_loss = self.critere(pred, true, indice, reduction='none')  # todo
            gradients = torch.autograd.grad(unreduced_loss.mean(), self.net.W(), retain_graph=True)
            with torch.no_grad():
                for w, vw, g in zip(self.net.W(), self.v_net.W(), gradients):
                    m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
                    vw.copy_(w - xi * (m + g + self.w_weight_decay * w))
                    w.grad = g
                for a, va in zip(self.net.A(), self.v_net.A()):
                    va.copy_(a)
        for r in range(0, self.args.world_size-1):
            if self.args.rank == r:
                pred, _, _, _, true, _ = self._process_one_batch_SCINet(next_set, next_data, self.v_net)
            dist.broadcast(pred.contiguous(), r)
            if self.args.rank == r+1:
                own_pred, _, _, _, true, _ = self._process_one_batch_SCINet(trn_set, trn_data, self.net)
                unreduced_loss = self.critere(own_pred, pred, indice, reduction='none')
                gradients = torch.autograd.grad(unreduced_loss.mean(), self.net.W())
                with torch.no_grad():
                    for w, vw, g in zip(self.net.W(), self.v_net.W(), gradients):
                        m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
                        vw.copy_(w - xi * (m + g + self.w_weight_decay * w))
                        w.grad = g
                    for a, va in zip(self.net.A(), self.v_net.A()):
                        va.copy_(a)
        return unreduced_loss

    def unrolled_backward(self, args_in, trn_data, trn_set, val_data, val_set, next_data, next_set, xi, w_optim, indice):
        """ Compute unrolled loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        # init config
        args = args_in
        # do virtual step (calc w`)
        unreduced_loss = self.virtual_step(trn_data, trn_set, next_data, next_set, xi, w_optim, indice)
        hessian = torch.zeros(args.batch_size, args.pred_len, trn_data[1].shape[-1]).to(self.device)
        if self.args.rank == 1:
            # calc unrolled loss
            pred, _, _, _, true, _ = self._process_one_batch_SCINet(val_set, val_data, self.v_net)
            loss = self.criterion(pred, true)
            # compute gradient
            v_W = list(self.v_net.W())
            dw = list(torch.autograd.grad(loss, v_W))
            hessian = self.compute_hessian(dw, trn_data, trn_set)
        elif self.args.rank == 0:
            dw_list = []
            for i in range(self.args.batch_size):
                dw_list.append(torch.autograd.grad(unreduced_loss[i], self.net.W(), retain_graph=(i != self.args.batch_size-1)))
        dist.broadcast(hessian, 1)
        da = None
        if self.args.rank == 0:
            pred, _, _, _, true, _ = self._process_one_batch_SCINet(trn_set, trn_data, self.v_net)
            assert pred.shape == hessian.shape
            pseudo_loss = (pred * hessian).sum()
            dw0 = torch.autograd.grad(pseudo_loss, self.v_net.W())
            if self.args.fourrier:
                weights = self.net.arch()[indice, :, :]
                d_weights = torch.zeros(self.args.batch_size, requires_grad=False)[:, None, None].to(self.device)
                for i in range(self.args.batch_size):
                    for a, b in zip(dw_list[i], dw0):
                        assert a.shape == b.shape
                        d_weights[i] += (a*b).sum()
                aux_loss = (d_weights * weights).sum()
                da = torch.autograd.grad(aux_loss, self.net.A())
                with torch.no_grad():
                    for a, d in zip(self.net.A(), da):
                        a.grad = d * xi * xi
            else:
                da = torch.zeros_like(self.net.arch).to(self.device)
                for i in range(self.args.batch_size):
                    for a, b in zip(dw_list[i], dw0):
                        da[indice[i]] += (a*b).sum()
                # update final gradient = dalpha - xi*hessian
                with torch.no_grad():
                    self.net.arch.grad = da * xi * xi
                # print(self.net.arch.grad[indice], da[indice])
        return unreduced_loss.mean(), da

    def compute_hessian(self, dw, trn_data, trn_set):
        """
        dw = dw` { L_val(alpha, w`, h`) }, dh = dh` { L_val(alpha, w`, h`) }
        w+ = w + eps_w * dw, h+ = h + eps_h * dh
        w- = w - eps_w * dw, h- = h - eps_h * dh
        hessian_w = (dalpha { L_trn(alpha, w+, h) } - dalpha { L_trn(alpha, w-, h) }) / (2*eps_w)
        hessian_h = (dalpha { L_trn(alpha, w, h+) } - dalpha { L_trn(alpha, w, h-) }) / (2*eps_h)
        eps_w = 0.01 / ||dw||, eps_h = 0.01  ||dh||
        """
        norm_w = torch.cat([w.view(-1) for w in dw]).norm()
        eps_w = 0.01 / norm_w
        trn_data[1].requires_grad = True

        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.W(), dw):
                p += eps_w * d
        pred, _, _, _, true, _ = self._process_one_batch_SCINet(trn_set, trn_data, self.net)
        loss = self.criterion(pred, true)
        dE_pos = torch.autograd.grad(loss, [trn_data[1]])[0][:, -self.args.pred_len:, :]

        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.W(), dw):
                p -= 2. * eps_w * d
        pred, _, _, _, true, _ = self._process_one_batch_SCINet(trn_set, trn_data, self.net)
        loss = self.criterion(pred, true)
        dE_neg = torch.autograd.grad(loss, [trn_data[1]])[0][:, -self.args.pred_len:, :]

        # recover w
        with torch.no_grad():
            for p, d in zip(self.net.W(), dw):
                p += eps_w * d

        hessian = (dE_pos - dE_neg) / (2. * eps_w)
        trn_data[1].requires_grad = False
        return hessian

    def _process_one_batch_SCINet(self, dataset_object, data, model):
        batch_x = data[0].double().to(self.device)
        batch_y = data[1].double()

        if self.args.stacks == 1:
            outputs = model(batch_x)
        elif self.args.stacks == 2:
            outputs, mid = model(batch_x)
        else:
            print('Error!')

        # if self.args.inverse:
        outputs_scaled = dataset_object.inverse_transform(outputs)
        if self.args.stacks == 2:
            mid_scaled = dataset_object.inverse_transform(mid)
        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        batch_y_scaled = dataset_object.inverse_transform(batch_y)

        if self.args.stacks == 1:
            return outputs, outputs_scaled, torch.tensor([0]).to(self.device), torch.tensor([0]).to(
                self.device), batch_y, batch_y_scaled
        elif self.args.stacks == 2:
            return outputs, outputs_scaled, mid, mid_scaled, batch_y, batch_y_scaled
        else:
            print('Error!')
