from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model import sigtemp
from models.architect_scinet import Architect_Scinet

from utils.tools import EarlyStopping, adjust_learning_rate, AverageMeter, MyDefiniteSampler
from utils.metrics import metric
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.functional import sigmoid
import torch.distributed as dist
from models.SCINet import SCINet

import os
import time

import warnings

warnings.filterwarnings('ignore')


class Exp_Scinet(Exp_Basic):
    def __init__(self, args):
        super(Exp_Scinet, self).__init__(args)

    def _build_model(self):
        train_data, _ = self._get_data(flag='train', samp=True)
        train_length = len(train_data)
        if self.args.features == 'S':
            in_dim = 1
        elif self.args.features == 'M':
            in_dim = 7
        else:
            print('Error!')

        model = SCINet(
            output_len=self.args.pred_len,
            input_len=self.args.seq_len,
            input_dim=in_dim,
            hid_size=self.args.hidden_size,
            num_stacks=self.args.stacks,
            num_levels=self.args.levels,
            concat_len=self.args.concat_len,
            groups=self.args.groups,
            kernel=self.args.kernel,
            dropout=self.args.dropout,
            single_step_output_One=self.args.single_step_output_One,
            positionalE=self.args.positionalEcoding,
            modified=True,
            RIN=self.args.RIN,
            device=self.device,
            train_length=train_length,
            args=self.args,
            fourrier=self.args.fourrier).double()
        print(model)
        self.arch = Architect_Scinet(model, self.device, self.args, self._select_criterion(self.args.loss))
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
        W_optim = optim.Adam(self.model.W(), lr=self.args.learning_rate)
        A_optim = optim.Adam(self.model.A(), self.args.A_lr, weight_decay=0)
        if self.args.fourrier:
            A_optim = optim.Adam(self.model.A(), self.args.A_lr, weight_decay=self.args.A_weight_decay)
        return W_optim, A_optim

    def _select_criterion(self, losstype):
        if losstype == "mse":
            criterion = nn.MSELoss()
        elif losstype == "mae":
            criterion = nn.L1Loss()
        else:
            criterion = nn.L1Loss()
        return criterion

    def vali(self, valid_data, valid_loader, criterion):
        self.model.eval()
        total_loss = []

        preds = []
        trues = []
        mids = []
        pred_scales = []
        true_scales = []
        mid_scales = []

        for i, val_data in enumerate(valid_loader):
            pred, pred_scale, mid, mid_scale, true, true_scale = self._process_one_batch_SCINet(
                valid_data, val_data)

            if self.args.stacks == 1:
                loss = criterion(pred.detach().cpu(), true.detach().cpu())

                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
                pred_scales.append(pred_scale.detach().cpu().numpy())
                true_scales.append(true_scale.detach().cpu().numpy())

            elif self.args.stacks == 2:
                loss = criterion(pred.detach().cpu(), true.detach().cpu()) + criterion(mid.detach().cpu(),
                                                                                       true.detach().cpu())

                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
                mids.append(mid.detach().cpu().numpy())
                pred_scales.append(pred_scale.detach().cpu().numpy())
                mid_scales.append(mid_scale.detach().cpu().numpy())
                true_scales.append(true_scale.detach().cpu().numpy())

            else:
                print('Error!')

            total_loss.append(loss)
        total_loss = np.average(total_loss)

        if self.args.stacks == 1:
            preds = np.array(preds)
            trues = np.array(trues)
            pred_scales = np.array(pred_scales)
            true_scales = np.array(true_scales)

            preds = preds.reshape((-1, preds.shape[-2], preds.shape[-1]))
            trues = trues.reshape((-1, trues.shape[-2], trues.shape[-1]))
            true_scales = true_scales.reshape((-1, true_scales.shape[-2], true_scales.shape[-1]))
            pred_scales = pred_scales.reshape((-1, pred_scales.shape[-2], pred_scales.shape[-1]))

            mae, mse, rmse, mape, mspe, corr = metric(preds, trues)
            maes, mses, rmses, mapes, mspes, corrs = metric(pred_scales, true_scales)
        elif self.args.stacks == 2:
            preds = np.array(preds)
            trues = np.array(trues)
            mids = np.array(mids)
            pred_scales = np.array(pred_scales)
            true_scales = np.array(true_scales)
            mid_scales = np.array(mid_scales)

            preds = preds.reshape((-1, preds.shape[-2], preds.shape[-1]))
            trues = trues.reshape((-1, trues.shape[-2], trues.shape[-1]))
            mids = mids.reshape((-1, mids.shape[-2], mids.shape[-1]))
            true_scales = true_scales.reshape((-1, true_scales.shape[-2], true_scales.shape[-1]))
            pred_scales = pred_scales.reshape((-1, pred_scales.shape[-2], pred_scales.shape[-1]))
            mid_scales = mid_scales.reshape((-1, mid_scales.shape[-2], mid_scales.shape[-1]))
            # print('test shape:', preds.shape, mids.shape, trues.shape)

            mae, mse, rmse, mape, mspe, corr = metric(mids, trues)
            maes, mses, rmses, mapes, mspes, corrs = metric(mid_scales, true_scales)
            mae, mse, rmse, mape, mspe, corr = metric(preds, trues)
            maes, mses, rmses, mapes, mspes, corrs = metric(pred_scales, true_scales)

        else:
            print('Error!')

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
        criterion = self._select_criterion(self.args.loss)
        if self.args.rank == 0 and ii == 0 and self.args.fourrier:
            logger.info("R{} cos{}, sin{}".format(self.args.rank, self.model.arch.cos, self.model.arch.sin))
            np.save(path + '/' + 'cos0.npy', self.model.arch.cos.detach().squeeze().cpu().numpy())
            np.save(path + '/' + 'sin0.npy', self.model.arch.sin.detach().squeeze().cpu().numpy())

        DA = []

        for epoch in range(self.args.train_epochs):
            DA.append([])
            iter_count = 0
            data_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, trn_data in enumerate(train_loader):
                try:
                    val_data = next(val_iter)
                except:
                    val_iter = iter(vali_loader)
                    val_data = next(val_iter)

                for j in range(len(trn_data)):
                    trn_data[j], val_data[j] = trn_data[j].double().to(self.device), val_data[j].double().to(self.device)
                iter_count += 1
                indice = train_loader.sampler.indice[data_count:data_count+self.args.batch_size]

                # # Update A
                # A_optim.zero_grad()
                # loss, da = self.arch.unrolled_backward(self.args, trn_data, train_data, val_data, vali_data, trn_data,
                #                                        train_data, self.args.unrolled, W_optim, indice)
                # DA[-1].append(0)
                # if self.args.rank == 0:
                #     for i, d in enumerate(da):
                #         DA[-1][-1] = (DA[-1][-1] * i + d.mean().cpu().item()) / (i+1)
                # A_optim.step()
                logger.info("R{} cp0".format(self.args.rank))
                # Update W
                W_optim.zero_grad()
                pred = torch.zeros(trn_data[1][:, -self.args.pred_len:, :].shape).to(self.device)
                if self.args.rank == 0:
                    pred, pred_scale, mid, mid_scale, true, true_scale = self._process_one_batch_SCINet(
                        train_data, trn_data)
                    loss = self.critere(pred, true, indice) + self.critere(mid, true, indice)
                    loss.backward()
                    logger.info("R{} cp1 {}".format(self.args.rank, loss))
                    W_optim.step()
                logger.info("R{} cp2".format(self.args.rank))
                for r in range(0, self.args.world_size - 1):
                    if self.args.rank == r:
                        pred, pred_scale, mid, mid_scale, true, true_scale = self._process_one_batch_SCINet(
                            train_data, trn_data)
                    logger.info("R{} cp3 {}".format(self.args.rank, pred))
                    dist.broadcast(pred.contiguous(), r)
                    logger.info("R{} cp4 {}".format(self.args.rank, pred))
                    if self.args.rank == r + 1:
                        own_pred, pred_scale, own_mid, mid_scale, true, true_scale = self._process_one_batch_SCINet(
                            train_data, trn_data)
                        logger.info("R{} cp5".format(self.args.rank))
                        loss1 = criterion(own_pred, true) + criterion(own_mid, true)

                        loss2 = criterion(own_pred, pred)  # todo: check shape of mid and add it to loss2
                        logger.info("R{} cp6 {} {}".format(self.args.rank, loss1, loss2))

                        loss = loss1 * (1-self.args.lambda_par) + loss2 * self.args.lambda_par
                        loss.backward()
                        W_optim.step()
                        logger.info("R{} cp7".format(self.args.rank))
                train_loss.append(loss.item())
                logger.info("R{} cp8".format(self.args.rank))
                if (i + 1) % 50 == 0:
                    logger.info("\tR{0} iters: {1}, epoch: {2} | loss: {3:.7f}".format(self.args.rank, i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    logger.info('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                data_count += self.args.batch_size

            if not self.args.fourrier:
                with torch.no_grad():
                    self.model.arch *= (1-self.args.A_weight_decay)

            logger.info("R{} Epoch: {} cost time: {}".format(self.args.rank, epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            logger.info("R{0} Epoch: {1}, Steps: {2} | Train Loss: {3:.7f} Vali Loss: {4:.7f} Test Loss: {5:.7f}".format(
                self.args.rank, epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            if not self.args.fourrier:
                logger.info("R{0} arch{1}".format(self.args.rank, self.model.arch.std()))
                if self.args.rank == 0 and ii == 0:
                    np.save(path + '/' + 'arch{}.npy'.format(epoch+1), self.model.arch.detach().squeeze().cpu().numpy())
                    # np.save(path + '/' + 'arch_factor{}.npy'.format(epoch), self.model.arch_1.detach().squeeze().cpu().numpy())
            elif self.args.rank == 0 and ii == 0:
                logger.info("R{} cos{}, sin{}".format(self.args.rank, self.model.arch.cos, self.model.arch.sin))
                np.save(path + '/' + 'cos{}.npy'.format(epoch+1), self.model.arch.cos.detach().squeeze().cpu().numpy())
                np.save(path + '/' + 'sin{}.npy'.format(epoch+1), self.model.arch.sin.detach().squeeze().cpu().numpy())
                np.save(path + '/' + 'da{}.npy'.format(epoch+1), np.array(DA[-1]))
            if epoch >= 2:
                early_stopping(vali_loss, self.model, path)
            flag = torch.tensor([1]) if early_stopping.early_stop else torch.tensor([0])
            flag = flag.to(self.device)
            flags = [torch.tensor([1]).to(self.device), torch.tensor([1]).to(self.device)]
            dist.all_gather(flags, flag)
            if flags[1].item() == 1:
                logger.info("Early stopping")
                break

            adjust_learning_rate(W_optim, epoch + 1, self.args)
            self.test(setting, logger)

        best_model_path = path + '/' + '{}_checkpoint.pth'.format(self.args.rank)
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, logger):
        test_data, test_loader = self._get_data(flag='test')

        self.model.eval()

        preds = []
        trues = []
        mids = []
        pred_scales = []
        true_scales = []
        mid_scales = []

        for i, trn_data in enumerate(test_loader):
            pred, pred_scale, mid, mid_scale, true, true_scale = self._process_one_batch_SCINet(
                test_data, trn_data)

            if self.args.stacks == 1:
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
                pred_scales.append(pred_scale.detach().cpu().numpy())
                true_scales.append(true_scale.detach().cpu().numpy())
            elif self.args.stacks == 2:
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
                mids.append(mid.detach().cpu().numpy())
                pred_scales.append(pred_scale.detach().cpu().numpy())
                mid_scales.append(mid_scale.detach().cpu().numpy())
                true_scales.append(true_scale.detach().cpu().numpy())

            else:
                print('Error!')

        if self.args.stacks == 1:
            preds = np.array(preds)
            trues = np.array(trues)

            pred_scales = np.array(pred_scales)
            true_scales = np.array(true_scales)

            preds = preds.reshape((-1, preds.shape[-2], preds.shape[-1]))
            trues = trues.reshape((-1, trues.shape[-2], trues.shape[-1]))
            true_scales = true_scales.reshape((-1, true_scales.shape[-2], true_scales.shape[-1]))
            pred_scales = pred_scales.reshape((-1, pred_scales.shape[-2], pred_scales.shape[-1]))

            mae, mse, rmse, mape, mspe, corr = metric(preds, trues)
            maes, mses, rmses, mapes, mspes, corrs = metric(pred_scales, true_scales)

        elif self.args.stacks == 2:
            preds = np.array(preds)
            trues = np.array(trues)
            mids = np.array(mids)

            pred_scales = np.array(pred_scales)
            true_scales = np.array(true_scales)
            mid_scales = np.array(mid_scales)

            preds = preds.reshape((-1, preds.shape[-2], preds.shape[-1]))
            trues = trues.reshape((-1, trues.shape[-2], trues.shape[-1]))
            true_scales = true_scales.reshape((-1, true_scales.shape[-2], true_scales.shape[-1]))
            pred_scales = pred_scales.reshape((-1, pred_scales.shape[-2], pred_scales.shape[-1]))
            mae, mse, rmse, mape, mspe, corr = metric(preds, trues)
            maes, mses, rmses, mapes, mspes, corrs = metric(pred_scales, true_scales)

        else:
            print('Error!')
        logger.info('R{} mse:{}, mae:{}'.format(self.args.rank, mse, mae))
        return mse, mae

    def _process_one_batch_SCINet(self, dataset_object, data):
        batch_x = data[0].double().to(self.device)
        batch_y = data[1].double()

        if self.args.stacks == 1:
            outputs = self.model(batch_x)
        elif self.args.stacks == 2:
            outputs, mid = self.model(batch_x)
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
            return outputs, outputs_scaled, torch.tensor([0]).to(self.device), torch.tensor([0]).to(self.device), batch_y, batch_y_scaled
        elif self.args.stacks == 2:
            return outputs, outputs_scaled, mid, mid_scaled, batch_y, batch_y_scaled
        else:
            print('Error!')

    def critere(self, pred, true, indice, reduction='mean'):
        if self.args.fourrier:
            weights = self.model.arch()[indice, :, :]
            weights = sigtemp(weights, self.args.temp) * self.args.sigmoid
        else:
            weights = self.model.arch[indice, :, :]
            weights = sigmoid(weights) * self.args.sigmoid
        # weights = self.model.normal_prob(self.model.arch)[indice[data_count:data_count + pred.shape[0]], :, :]
        if reduction != 'mean':
            crit = nn.MSELoss(reduction=reduction)
            return (crit(pred, true) * weights).mean(dim=(-1, -2))
        else:
            crit = nn.MSELoss(reduction=reduction)
            return (crit(pred, true) * weights).mean()



