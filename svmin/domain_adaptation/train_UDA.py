import os
import sys
from pathlib import Path

import os.path as osp
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch import nn
from torchvision.utils import make_grid
from tqdm import tqdm

from advent.model.discriminator import get_fc_discriminator
from advent.utils.func import adjust_learning_rate, adjust_learning_rate_discriminator
from advent.utils.func import loss_calc, bce_loss
from advent.utils.loss import entropy_loss
from advent.utils.func import prob_2_entropy
from advent.utils.viz_segmask import colorize_mask

def train_domain_adaptation(model, trainloader, targetloader, cfg):
    if cfg.TRAIN.DA_METHOD == 'SVMin':
        train_SVMin(model, trainloader, targetloader, cfg)
    else:
        raise NotImplementedError(f"Not yet supported DA method {cfg.TRAIN.DA_METHOD}")

def train_SVMin(model, trainloader, targetloader, cfg):
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)

    trainloader_iter = enumerate(trainloader)
    targetloader_iter = enumerate(targetloader)
    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP + 1)):
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter, cfg)

        _, batch = trainloader_iter.__next__()
        images_source, labels, _, _ = batch
        pred_src_aux, pred_src = model(images_source.cuda(device))
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = interp(pred_src_aux)
            loss_seg_src_aux = loss_calc(pred_src_aux, labels, device)
        else:
            loss_seg_src_aux = 0
        pred_src_main = interp(pred_src)
        loss_seg_src_main = loss_calc(pred_src_main, labels, device)
        loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux)
        loss.backward()

        _, batch = targetloader_iter.__next__()
        images, _, _, _ = batch
        with torch.no_grad():
            pred_trg_aux, pred_trg_main = model(images.cuda(device))
        scale_ratio = np.random.randint(100.0*cfg.TRAIN.SCALING_RATIO[0], 100.0*cfg.TRAIN.SCALING_RATIO[1]) / 100.0
        scaled_size_target = (round(input_size_target[1] * scale_ratio / 8) * 8, round(input_size_target[0] * scale_ratio / 8) * 8)
        print('\n### scaled_size_target : %.0f * %.0f ' % scaled_size_target)
        interp_target_sc = nn.Upsample(size=scaled_size_target, mode='bilinear', align_corners=True)
        images_sc = interp_target_sc(images)
        pred_trg_aux_sc, pred_trg_main_sc = model(images_sc.cuda(device))
        interp_target_sc2ori = nn.Upsample(size = (pred_trg_main.shape[-2], pred_trg_main.shape[-1]), mode='bilinear', align_corners=True)
        pred_trg_main_sc2ori = interp_target_sc2ori(pred_trg_main_sc)
        out_trg_main_sc2ori = F.softmax(pred_trg_main_sc2ori)
        out_trg_main = F.softmax(pred_trg_main)
        loss_svmin = l1_loss(out_trg_main_sc2ori, out_trg_main)
        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_aux_sc2ori = interp_target_sc2ori(pred_trg_aux_sc)
            out_trg_aux_sc2ori = F.softmax(pred_trg_aux_sc2ori)
            out_trg_aux = F.softmax(pred_trg_aux)
            loss_svmin_aux = l1_loss(out_trg_aux_sc2ori, out_trg_aux)
        else:
            loss_svmin_aux = 0

        loss = (cfg.TRAIN.LAMBDA_SVMIN * loss_svmin+
                cfg.TRAIN.LAMBDA_SVMIN * loss_svmin_aux)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        current_losses = {'loss_seg_src_main': loss_seg_src_main,
                          'loss_seg_src_aux': loss_seg_src_aux,
                          'loss_svmin': loss_svmin,
                          'loss_svmin_aux': loss_svmin_aux}
        print_losses(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')

            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()

def l1_loss(input, target):
    loss = torch.abs(input - target)
    loss = torch.mean(loss)
    return loss

def draw_in_tensorboard(writer, images, i_iter, pred_main, num_classes, type_):
    grid_image = make_grid(images[:3].clone().cpu().data, 3, normalize=True)
    writer.add_image(f'Image - {type_}', grid_image, i_iter)

    grid_image = make_grid(torch.from_numpy(np.array(colorize_mask(np.asarray(
        np.argmax(F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0),
                  axis=2), dtype=np.uint8)).convert('RGB')).transpose(2, 0, 1)), 3,
                           normalize=False, range=(0, 255))
    writer.add_image(f'Prediction - {type_}', grid_image, i_iter)

    output_sm = F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0)
    output_ent = np.sum(-np.multiply(output_sm, np.log2(output_sm)), axis=2,
                        keepdims=False)
    grid_image = make_grid(torch.from_numpy(output_ent), 3, normalize=True,
                           range=(0, np.log2(num_classes)))
    writer.add_image(f'Entropy - {type_}', grid_image, i_iter)

def print_losses(current_losses, i_iter):
    list_strings = []
    for loss_name, loss_value in current_losses.items():
        list_strings.append(f'{loss_name} = {to_numpy(loss_value):.3f} ')
    full_string = ' '.join(list_strings)
    tqdm.write(f'iter = {i_iter} {full_string}')

def log_losses_tensorboard(writer, current_losses, i_iter):
    for loss_name, loss_value in current_losses.items():
        writer.add_scalar(f'data/{loss_name}', to_numpy(loss_value), i_iter)

def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()
