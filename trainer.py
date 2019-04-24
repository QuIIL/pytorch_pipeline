
import numpy as np
import matplotlib.pyplot as plt

import shutil
import argparse
import os
import json
import random
import warnings
from termcolor import colored
import pandas as pd
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as data

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage
from tensorboardX import SummaryWriter

import imgaug # https://github.com/aleju/imgaug
from imgaug import augmenters as iaa

import misc
import dataset
from model.net import DenseNet
from config import Config

####
class Trainer(Config):
    ####
    def view_dataset(self, mode='train'):
        train_pairs, valid_pairs = dataset.prepare_smhtma_data()
        if mode == 'train':
            train_augmentors = self.train_augmentors()
            ds = dataset.DatasetSerial(train_pairs, has_aux=self.attent,
                            shape_augs=iaa.Sequential(train_augmentors[0]),
                            input_augs=iaa.Sequential(train_augmentors[1]))
        else:
            infer_augmentors = self.infer_augmentors() # HACK
            ds = dataset.DatasetSerial(valid_pairs, has_aux=False,
                            shape_augs=iaa.Sequential(infer_augmentors)[0])
        dataset.visualize(ds, 4)
        return
    ####
    def train_step(self, net, batch, optimizer, device):
        net.train() # train mode

        imgs_cpu, true_cpu, nucs_cpu = batch
        imgs_cpu = imgs_cpu.permute(0, 3, 1, 2) # to NCHW

        # push data to GPUs
        imgs = imgs_cpu.to(device).float()
        true = true_cpu.to(device).long() # not one-hot
        nucs = nucs_cpu.to(device).long()

        # -----------------------------------------------------------
        net.zero_grad() # not rnn so not accumulate

        logit, aux_logit = net(imgs) # forward
        prob = F.softmax(logit, dim=-1)

        # has built-int log softmax so accept logit
        loss = F.cross_entropy(logit, true, reduction='mean')
        pred = torch.argmax(prob, dim=-1)
        acc  = torch.mean((pred == true).float()) # batch accuracy

        #
        aux_prob = F.softmax(aux_logit, dim=1)
        aux_loss = F.cross_entropy(aux_logit, nucs, reduction='mean')
        
        class_loss = loss
        loss = loss + aux_loss
        # gradient update
        loss.backward()
        optimizer.step()

        # -----------------------------------------------------------
        return dict(class_loss=class_loss.item(),
                    class_acc=acc.item(),
                    seg_loss=aux_loss.item(), 
                    seg_imgs=[
                        imgs_cpu.numpy(), 
                        nucs_cpu.numpy(), 
                        aux_prob.detach().cpu().numpy()]
                    )
    ####
    def infer_step(self, net, batch, device):
        net.eval() # infer mode

        imgs, true = batch
        imgs = imgs.permute(0, 3, 1, 2) # to NCHW

        # push data to GPUs and convert to float32
        imgs = imgs.to(device).float()
        true = true.to(device).long() # not one-hot

        # -----------------------------------------------------------
        with torch.no_grad(): # dont compute gradient
            logit = net(imgs)[0] # HACK
            prob = nn.functional.softmax(logit, dim=-1)
            return dict(prob=prob.cpu().numpy(), 
                        true=true.cpu().numpy())
    ####
    def run_once(self, fold_idx):
        
        log_dir = '%s/%02d/' % (self.log_dir, fold_idx)

        misc.check_manual_seed(self.seed)
        train_pairs, valid_pairs = dataset.prepare_smhtma_data(fold_idx)
        # train_pairs, valid_pairs = dataset.prepare_colon_data(fold_idx)

        # --------------------------- Dataloader

        train_augmentors = self.train_augmentors()
        train_dataset = dataset.DatasetSerial(train_pairs[:], has_aux=self.attent,
                        shape_augs=iaa.Sequential(train_augmentors[0]),
                        input_augs=iaa.Sequential(train_augmentors[1]))

        infer_augmentors = self.infer_augmentors() # HACK at has_aux
        infer_dataset = dataset.DatasetSerial(valid_pairs[:], has_aux=False,
                        shape_augs=iaa.Sequential(infer_augmentors[0]))

        train_loader = data.DataLoader(train_dataset, 
                                num_workers=self.nr_procs_train, 
                                batch_size=self.train_batch_size, 
                                shuffle=True, drop_last=True)

        valid_loader = data.DataLoader(infer_dataset, 
                                num_workers=self.nr_procs_valid, 
                                batch_size=self.infer_batch_size, 
                                shuffle=True, drop_last=False)

        # --------------------------- Training Sequence

        if self.logging:
            misc.check_log_dir(log_dir)

        device = 'cuda'

        # networks
        input_chs = 3
        if self.attent:
            if self.guide_mode == 'concat':
                input_chs = 4
    
        net = DenseNet(input_chs, self.nr_classes)

        # load pre-trained models
        if self.load_network:
            saved_state = torch.load(self.save_net_path)
            net.load_state_dict(saved_state)

        net = torch.nn.DataParallel(net).to(device)

        # optimizers
        optimizer = optim.Adam(net.parameters(), lr=self.init_lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, self.lr_steps)

        #
        trainer = Engine(lambda engine, batch: self.train_step(net, batch, optimizer, device))
        inferer = Engine(lambda engine, batch: self.infer_step(net, batch, device))

        train_output = ['class_loss', 'class_acc', 'seg_loss']
        infer_output = ['prob', 'true']
        ##

        if self.logging:
            checkpoint_handler = ModelCheckpoint(log_dir, self.chkpts_prefix, 
                                            save_interval=1, n_saved=30, require_empty=False)
            # adding handlers using `trainer.add_event_handler` method API
            trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=checkpoint_handler,
                                    to_save={'net': net}) 

        timer = Timer(average=True)
        timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                            pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)
        timer.attach(inferer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                            pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

        # attach running average metrics computation
        # decay of EMA to 0.95 to match tensorpack default
        RunningAverage(alpha=0.95, output_transform=lambda x: x['class_loss']).attach(trainer, 'class_loss')
        RunningAverage(alpha=0.95, output_transform=lambda x: x['class_acc']).attach(trainer, 'class_acc')
        RunningAverage(alpha=0.95, output_transform=lambda x: x['seg_loss']).attach(trainer, 'seg_loss')

        # attach progress bar
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=['class_loss'])
        pbar.attach(inferer)

        # adding handlers using `trainer.on` decorator API
        @trainer.on(Events.EXCEPTION_RAISED)
        def handle_exception(engine, e):
            if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
                engine.terminate()
                warnings.warn('KeyboardInterrupt caught. Exiting gracefully.')
                checkpoint_handler(engine, {'net_exception': net})
            else:
                raise e

        # writer for tensorboard logging
        if self.logging:
            writer = SummaryWriter(log_dir=log_dir)
            json_log_file = log_dir + '/stats.json'
            with open(json_log_file, 'w') as json_file:
                json.dump({}, json_file) # create empty file

        @trainer.on(Events.EPOCH_STARTED)
        def log_lrs(engine):
            if self.logging:
                lr = float(optimizer.param_groups[0]['lr'])
                writer.add_scalar("lr", lr, engine.state.epoch)
            # advance scheduler clock
            scheduler.step()

        ####
        def update_logs(output, epoch, prefix, color):
            # print values and convert
            max_length = len(max(output.keys(), key=len))
            for metric in output:
                key = colored(prefix + '-' + metric.ljust(max_length), color)
                print('------%s : ' % key, end='')
                if metric != 'conf_mat':
                    print('%0.7f' % output[metric])
                else:
                    conf_mat = output['conf_mat'] # use pivot to turn back
                    conf_mat_df = pd.DataFrame(conf_mat)
                    conf_mat_df.index.name = 'True'
                    conf_mat_df.columns.name = 'Pred'
                    output['conf_mat'] = conf_mat_df
                    print('\n', conf_mat_df)
            if 'train' in prefix:
                lr = float(optimizer.param_groups[0]['lr'])
                key = colored(prefix + '-' + 'lr'.ljust(max_length), color)
                print('------%s : %0.7f' % (key, lr))

            if not self.logging:
                return

            # create stat dicts
            stat_dict = {}
            for metric in output:
                if metric != 'conf_mat':
                    metric_value = output[metric] 
                else:
                    conf_mat_df = output['conf_mat'] # use pivot to turn back
                    conf_mat_df = conf_mat_df.unstack().rename('value').reset_index()
                    conf_mat_df = pd.Series({'conf_mat' : conf_mat}).to_json(orient='records')
                    metric_value = conf_mat_df
                stat_dict['%s-%s' % (prefix, metric)] = metric_value

            # json stat log file, update and overwrite
            with open(json_log_file) as json_file:
                json_data = json.load(json_file)

            current_epoch = str(epoch)
            if current_epoch in json_data:
                old_stat_dict = json_data[current_epoch]
                stat_dict.update(old_stat_dict)
            current_epoch_dict = {current_epoch : stat_dict}
            json_data.update(current_epoch_dict)

            with open(json_log_file, 'w') as json_file:
                json.dump(json_data, json_file)

            # log values to tensorboard
            for metric in output:
                if metric != 'conf_mat':
                    writer.add_scalar(prefix + '-' + metric, output[metric], current_epoch)

        import cv2
        import matplotlib.pyplot as plt
        cmap_jet = plt.get_cmap('jet')
        cmap_vir = plt.get_cmap('viridis')

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_train_running_results(engine):
            """
            running training measurement
            """
            training_ema_output = engine.state.metrics #
            update_logs(training_ema_output, engine.state.epoch, prefix='train-ema', color='green')

            imgs, nucs, segs = engine.state.output['seg_imgs'] # NCHW
            imgs = np.transpose(imgs, [0, 2, 3, 1]).astype('float32') / 255.0
            nucs = nucs.astype('float32')
            segs = np.transpose(segs, [0, 2, 3, 1])[...,1]

            imgs = np.concatenate([imgs[0], imgs[1]], axis=0)
            nucs = cmap_vir(np.concatenate([nucs[0], nucs[1]], axis=0))[...,:3]
            segs = cmap_jet(np.concatenate([segs[0], segs[1]], axis=0))[...,:3]
            imgs = cv2.resize(imgs, (0, 0),  fx=1/8 , fy=1/8 , interpolation=cv2.INTER_NEAREST)
            tracked_images = np.concatenate([imgs, nucs, segs], axis=1)
            # plt.imshow(tracked_images)
            # plt.show()
            tracked_images = np.expand_dims(tracked_images, axis=0) # fake NCHW
            tracked_images = np.transpose(tracked_images, [0, 3, 1, 2])
            tracked_images = (tracked_images * 255).astype('uint8')
            writer.add_image('train/Image', tracked_images, engine.state.epoch)
        ####
        def get_init_accumulator(output_names):
            return {metric : [] for metric in output_names}

        def process_accumulated_output(output):
            #
            def uneven_seq_to_np(seq, batch_size=self.infer_batch_size):
                item_count = batch_size * (len(seq) - 1) + len(seq[-1])
                cat_array = np.zeros((item_count,) + seq[0][0].shape, seq[0].dtype)
                # BUG: odd len even
                for idx in range(0, len(seq)-1):
                    cat_array[idx   * batch_size : 
                            (idx+1) * batch_size] = seq[idx] 
                cat_array[(idx+1) * batch_size:] = seq[-1]
                return cat_array
            #
            prob = uneven_seq_to_np(output['prob'])
            true = uneven_seq_to_np(output['true'])
            # threshold then get accuracy
            pred = np.argmax(prob, axis=-1)
            acc = np.mean(pred == true)
            # confusion matrix
            conf_mat = confusion_matrix(true, pred, 
                                labels=np.arange(self.nr_classes))
            #
            proc_output = dict(acc=acc, conf_mat=conf_mat)
            return proc_output

        @trainer.on(Events.EPOCH_COMPLETED)
        def infer_valid(engine):
            """
            inference measurement
            """
            inferer.accumulator = get_init_accumulator(infer_output)
            inferer.run(valid_loader)
            output_stat = process_accumulated_output(inferer.accumulator)
            update_logs(output_stat, engine.state.epoch, prefix='valid', color='red')

        @inferer.on(Events.ITERATION_COMPLETED)    
        def accumulate_outputs(engine):
            batch_output = engine.state.output
            for key, item in batch_output.items():
                engine.accumulator[key].extend([item])
        ###
            
        # Setup is done. Now let's run the training
        trainer.run(train_loader, self.nr_epochs)
        return
    ####
    def run(self):
        if self.cross_valid:
            for fold_idx in range (0, trainer.nr_fold):
                trainer.run_once(fold_idx)
        else:
            self.run_once(self.fold_idx)
        return

####
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--view', help='view dataset', action='store_true')
    args = parser.parse_args()

    trainer = Trainer()
    if args.view:
        trainer.view_dataset()
        exit()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    trainer.run()