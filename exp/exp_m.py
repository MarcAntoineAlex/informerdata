from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack
from models.architect import Architect

from utils.tools import EarlyStopping, adjust_learning_rate, AverageMeter, MyDefiniteSampler
from utils.metrics import metric
import random
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.functional import sigmoid
import torch.distributed as dist

import os
import time

import warnings

warnings.filterwarnings('ignore')


class Exp_M_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_M_Informer, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'informer': Informer,
            'informerstack': InformerStack,
        }
        if self.args.model == 'informer' or self.args.model == 'informerstack':
            e_layers = self.args.e_layers if self.args.model == 'informer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in,
                self.args.c_out,
                self.args.seq_len,
                self.args.label_len,
                self.args.pred_len,
                self.args.factor,
                self.args.d_model,
                self.args.n_heads,
                e_layers,  # self.args.e_layers,
                self.args.d_layers,
                self.args.d_ff,
                self.args.dropout,
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device,
            ).float()
        else:
            raise NotImplementedError
        # something

        self.arch = Architect(model, self.device, self.args, self._select_criterion())
        return model

    def _get_data(self, flag, samp=False):
        args = self.args

        data_dict = {
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_minute,
            'ETTm2': Dataset_ETT_minute,
            'WTH': Dataset_Custom,
            'ECL': Dataset_Custom,
            'Solar': Dataset_Custom,
            'custom': Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed != 'timeF' else 1

        if flag == 'test':
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
        elif flag == 'pred':
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            freq = args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        sampler = None
        if samp:
            indices = list(range(len(data_set)))
            print("Lenth of data set : ", len(data_set))
            sampler = MyDefiniteSampler(indices, self.device)
            shuffle_flag = False

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            sampler=sampler)

        return data_set, data_loader

    def _select_optimizer(self):
        W_optim = optim.Adam(self.model.W(), 0.00005, weight_decay=self.args.w_weight_decay)
        A_optim = optim.Adam(self.model.A(), self.args.A_lr, betas=(0.5, 0.999),
                             weight_decay=0.)
        return W_optim, A_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, val_d in enumerate(vali_loader):
            pred, true = self._process_one_batch(val_d)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, ii, setting, logger):
        train_data, train_loader = self._get_data(flag='train', samp=True)
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.path, str(ii))
        try:
            os.mkdir(path)
        except FileExistsError:
            pass
        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, rank=self.args.rank, logger=logger)

        W_optim, A_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            data_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, trn_data in enumerate(train_loader):
                true_list = [torch.zeros_like(trn_data[1]).to(self.device) for _ in range(2)]
                if self.args.rank == 0:
                    dist.all_gather(true_list, trn_data[1].to(self.device))
                elif self.args.rank == 1:
                    dist.all_gather(true_list, trn_data[1].to(self.device))
                if torch.abs(true_list[0] - true_list[1]).max().item() != 0:
                    logger.info("DANGER 176")

                try:
                    val_data = next(val_iter)
                except:
                    val_iter = iter(vali_loader)
                    val_data = next(val_iter)

                for j in range(len(trn_data)):
                    trn_data[j], val_data[j] = trn_data[j].float().to(self.device), val_data[j].float().to(self.device)
                iter_count += 1

                # A_optim.zero_grad()
                # loss = self.arch.unrolled_backward(self.args, trn_data, val_data, trn_data, W_optim.param_groups[0]['lr'], W_optim, data_count)
                # A_optim.step()
                W_optim.zero_grad()
                pred = torch.zeros(trn_data[1][:, -self.args.pred_len:, :].shape).to(self.device)
                if self.args.rank == 0:
                    pred, true = self._process_one_batch(trn_data)
                    loss = self.critere(pred, true, data_count, criterion)
                    # loss = criterion(pred, true)  # todo: check
                    loss.backward()
                    W_optim.step()
                for r in range(0, self.args.world_size - 1):
                    if self.args.rank == r:
                        pred, true = self._process_one_batch(trn_data)
                    dist.broadcast(pred.contiguous(), r)
                    if self.args.rank == r + 1:
                        own_pred, true = self._process_one_batch(trn_data)
                        loss1 = criterion(own_pred, true)
                        loss2 = criterion(own_pred, pred)
                        loss = loss1 + loss2 * self.args.lambda_par
                        loss.backward()
                        W_optim.step()
                train_loss.append(loss.item())

                if (i + 1) % 50 == 0:
                    logger.info("\tR{0} iters: {1}, epoch: {2} | loss: {3:.7f}".format(self.args.rank, i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    logger.info('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()


                data_count += self.args.batch_size

            logger.info("R{} Epoch: {} cost time: {}".format(self.args.rank, epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            logger.info("R{0} Epoch: {1}, Steps: {2} | Train Loss: {3:.7f} Vali Loss: {4:.7f} Test Loss: {5:.7f}".format(
                self.args.rank, epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            logger.info("R{0} arch{1}".format(self.args.rank, self.model.arch.std()))
            early_stopping(vali_loss, self.model, path)

            flag = torch.tensor([1]) if early_stopping.early_stop else torch.tensor([0])
            flag = flag.to(self.device)
            flags = [torch.tensor([1]).to(self.device), torch.tensor([1]).to(self.device)]
            dist.all_gather(flags, flag)
            if flags[1].item() == 1 and flags[0].item() == 1:
                logger.info("Early stopping")
                break

            # adjust_learning_rate(W_optim, epoch + 1, self.args)
            self.test(setting, logger)

        best_model_path = path + '/' + '{}_checkpoint.pth'.format(self.args.rank)
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, logger):
        test_data, test_loader = self._get_data(flag='test')

        self.model.eval()

        preds = []
        trues = []

        for i, test_d in enumerate(test_loader):
            pred, true = self._process_one_batch(test_d)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape((-1, preds.shape[-2], preds.shape[-1]))
        trues = trues.reshape((-1, trues.shape[-2], trues.shape[-1]))

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        logger.info('R{} mse:{}, mae:{}'.format(self.args.rank, mse, mae))

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return mse, mae

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()

        preds = []

        for i, pred_d in enumerate(pred_loader):
            pred, true = self._process_one_batch(pred_d)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape((-1, preds.shape[-2], preds.shape[-1]))

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return

    def _process_one_batch(self, data):
        batch_x = data[0].float().to(self.device)
        batch_y = data[1].float().to(self.device)

        batch_x_mark = data[2].float().to(self.device)
        batch_y_mark = data[3].float().to(self.device)

        # decoder input
        if self.args.padding == 0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float().to(self.device)
        elif self.args.padding == 1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float().to(self.device)
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        return outputs, batch_y

    def critere(self, pred, true, data_count, criterion, reduction='mean'):
        weights = self.model.arch[data_count:data_count + pred.shape[0]]
        weights = sigmoid(weights) * 2
        if reduction != 'mean':
            crit = nn.MSELoss(reduction=reduction)
            return crit(pred * weights, true * weights).mean(dim=(-1, -2))
        else:
            crit = nn.MSELoss(reduction=reduction)
            return (crit(pred, true) * weights).mean()



