'''
The default exp_name is tmp. Change it before formal training! isic2018 PH2 DMF SKD
nohup python -u multi_train_adapt.py --exp_name test --config_yml Configs/multi_train_local.yml --model MedFormer --batch_size 16 --adapt_method False --num_domains 1 --dataset PH2  --k_fold 4 > 4MedFormer_PH2.out 2>&1 &
'''
import numpy as np
import argparse
from sqlite3 import adapt
import yaml
import os, time
from datetime import datetime
import cv2
import wandb
import torch

import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
import medpy.metric.binary as metrics

from Datasets.create_dataset import *
from Datasets.rankmatch import *
from Datasets.transform import normalize
from Utils.losses import dice_loss, corr_loss
from Utils.pieces import DotDict
from Utils.functions import fix_all_seed, segmentation_metrics

from Models.Transformer.SwinUnetUni import SwinUnet
from itertools import cycle

torch.cuda.empty_cache()

def save_checkpoint(model, optimizer, scheduler, epoch, best_metrics, config, args, is_best=False):
    """Save checkpoint to disk and wandb"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_metrics': best_metrics,
        'config': config,
        'args': args
    }
    
    # Save latest checkpoint
    checkpoint_dir = f"checkpoints/{config.data.name}/{args.exp}_{config.data.supervised_ratio}/fold{args.fold}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_path = f"{checkpoint_dir}/latest.pth"
    torch.save(checkpoint, latest_path)
    
    # Save best checkpoint if needed
    if is_best:
        best_path = f"{checkpoint_dir}/best.pth"
        torch.save(checkpoint, best_path)
    
    # Save to wandb
    artifact = wandb.Artifact(
        name=f"{args.exp}_{config.data.supervised_ratio}_fold{args.fold}_latest",
        type="model",
        description=f"Latest checkpoint for {args.exp} with {config.data.supervised_ratio} supervised ratio, fold {args.fold}"
    )
    artifact.add_file(latest_path)
    wandb.log_artifact(artifact)

def load_checkpoint_from_wandb(model, optimizer, scheduler, artifact_name, device='cuda'):
    """Load checkpoint from wandb"""
    api = wandb.Api()
    artifact = api.artifact(artifact_name)
    artifact_dir = artifact.download()
    
    checkpoint = torch.load(f"{artifact_dir}/latest.pth", map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['best_metrics']

def main(config):
    
    # wandb.init(project="SkinSeg", name=f"RankMatch_fold{config.fold}", config=config)
    wandb.init(mode="disabled")
    dataset = get_dataset(config, img_size=config.data.img_size, 
                                                    supervised_ratio=config.data.supervised_ratio, 
                                                    train_aug=config.data.train_aug,
                                                    k=config.fold,
                                                    ulb_dataset=SemiDataset,
                                                    lb_dataset=SkinDataset2)

    l_train_loader = torch.utils.data.DataLoader(dataset['lb_dataset'],
                                                batch_size=config.train.l_batchsize,
                                                shuffle=True,
                                                num_workers=config.train.num_workers,
                                                pin_memory=True,
                                                drop_last=False)
    u_train_loader = torch.utils.data.DataLoader(dataset['ulb_dataset'],
                                                batch_size=config.train.u_batchsize,
                                                shuffle=True,
                                                num_workers=config.train.num_workers,
                                                pin_memory=True,
                                                drop_last=False)
    val_loader = torch.utils.data.DataLoader(dataset['val_dataset'],
                                                batch_size=config.test.batch_size,
                                                shuffle=False,
                                                num_workers=config.test.num_workers,
                                                pin_memory=True,
                                                drop_last=False)
    test_loader = torch.utils.data.DataLoader(dataset['val_dataset'],
                                                batch_size=config.test.batch_size,
                                                shuffle=False,
                                                num_workers=config.test.num_workers,
                                                pin_memory=True,
                                                drop_last=False)
    train_loader = {'l_loader':l_train_loader, 'u_loader':u_train_loader}
    print(len(u_train_loader), len(l_train_loader))

    model = SwinUnet(img_size=config.data.img_size)

    total_trainable_params = sum(
                    p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print('{}M total parameters'.format(total_params/1e6))
    print('{}M total trainable parameters'.format(total_trainable_params/1e6))
    
    model = model.cuda()
    criterion = [nn.BCELoss(), dice_loss, corr_loss]

    # Initialize optimizer and scheduler
    if config.train.optimizer.mode == 'adam':
        print('choose wrong optimizer')
    elif config.train.optimizer.mode == 'adamw':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=float(config.train.optimizer.adamw.lr),
                              weight_decay=float(config.train.optimizer.adamw.weight_decay))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # Initialize training state
    start_epoch = 0
    best_metrics = {
        'val_dice': 0.0,
        'val_iou': 0.0,
        'val_acc': 0.0,
        'val_sen': 0.0,
        'val_spe': 0.0
    }

    # Resume from checkpoint if specified
    if args.resume is not None:
        print(f"Resuming from checkpoint: {args.resume}")
        if args.resume.startswith('wandb://'):
            # Load from wandb
            artifact_name = args.resume.replace('wandb://', '')
            print(f"Loading checkpoint from wandb artifact: {artifact_name}")
            start_epoch, best_metrics = load_checkpoint_from_wandb(model, optimizer, scheduler, artifact_name)
        else:
            # Load from local file
            checkpoint = torch.load(args.resume, map_location='cuda')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_metrics = checkpoint['best_metrics']
        print(f"Resumed from epoch {start_epoch-1}")

    # only test
    if config.test.only_test == True:
        test(config, model, config.test.test_model_dir, test_loader, criterion)
    else:
        train_val(config, model, train_loader, val_loader, criterion, optimizer, scheduler, start_epoch, best_metrics, args)
        test(config, model, best_model_dir, test_loader, criterion)

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))
def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)

# =======================================================================================================
def train_val(config, model, train_loader, val_loader, criterion, optimizer, scheduler, start_epoch, best_metrics, args):
    # ---------------------------------------------------------------------------
    # Training and Validating
    #----------------------------------------------------------------------------
    epochs = config.train.num_epochs
    max_iou = 0 # use for record best model
    max_dice = 0 # use for record best model
    best_epoch = start_epoch # use for recording the best epoch
    
    # Save initial checkpoint if starting from beginning
    if start_epoch == 0:
        save_checkpoint(model, optimizer, scheduler, 0, best_metrics, config, args)
    
    for epoch in range(start_epoch, epochs):
        start = time.time()
        # ----------------------------------------------------------------------
        # train
        # ---------------------------------------------------------------------
        model.train()
        dice_train_sum= 0
        iou_train_sum = 0
        loss_train_sum = 0
        num_train = 0
        iter = 0
        bce_sup_loss_sum = 0
        dice_sup_loss_sum = 0
        bce_unsup_loss_sum = 0
        dice_unsup_loss_sum = 0
        corr_unsup_loss_sum = 0
        source_dataset = zip(cycle(train_loader['l_loader']), train_loader['u_loader'], train_loader['u_loader'])
        for i, (batch,
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2),
                (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, ignore_mask_mix, _, _)) in enumerate(source_dataset):
            
            torch.cuda.empty_cache()
            img_x, mask_x = batch['image'].cuda(), batch['label'].cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2, ignore_mask = img_u_s1.cuda(), img_u_s2.cuda(), ignore_mask.cuda()
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()
            
            sup_batch_len = img_x.shape[0]
            unsup_batch_len = img_u_w.shape[0]
            with torch.no_grad():
                model.eval()

                feat_u_w_mix = model(img_u_w_mix).detach()
                pred_u_w_mix = torch.sigmoid(feat_u_w_mix)
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)
                _, H, W =conf_u_w_mix.shape
                _, _, h, w = feat_u_w_mix.shape

            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
                img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]
            
            model.train()

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

            feats, feats_fp = model(torch.cat((img_x, img_u_w)), True)
            preds = torch.sigmoid(feats)
            preds_fp = torch.sigmoid(feats_fp)
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])
            _, feat_u_w = feats.split([num_lb, num_ulb])

            pred_u_w_fp = preds_fp[num_lb:]

            feats_u_s = model(torch.cat((img_u_s1, img_u_s2)))
            preds_u_s = torch.sigmoid(feats_u_s)
            pred_u_s1, pred_u_s2 = preds_u_s.chunk(2)
            feat_u_s1, feat_u_s2 = feats_u_s.chunk(2)

            pred_u_w = pred_u_w.detach()
            feat_u_w = feat_u_w.detach()
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)

            feat_u_w = F.interpolate(feat_u_w, size=(H, W), mode="bilinear", align_corners=True)
            feat_u_w_mix = F.interpolate(feat_u_w_mix, size=(H, W), mode="bilinear", align_corners=True)

            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1, feat_u_w_cutmixed1 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone(), feat_u_w.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2, feat_u_w_cutmixed2 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone(), feat_u_w.clone()
            
            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask_mix[cutmix_box1 == 1]


            cutmix_box1 = cutmix_box1.unsqueeze(1).expand(feat_u_w.shape)
            feat_u_w_cutmixed1 = torch.where(cutmix_box1 == 1, feat_u_w_mix, feat_u_w_cutmixed1)
            feat_u_w_cutmixed1 = F.interpolate(feat_u_w_cutmixed1, size=(h, w), mode="bilinear", align_corners=True)

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask_mix[cutmix_box2 == 1]
            cutmix_box2 = cutmix_box2.unsqueeze(1).expand(feat_u_w.shape)
            feat_u_w_cutmixed2 = torch.where(cutmix_box2 == 1, feat_u_w_mix, feat_u_w_cutmixed2)
            feat_u_w_cutmixed2 = F.interpolate(feat_u_w_cutmixed2, size=(h, w), mode="bilinear", align_corners=True)


            losses_l = []
            for function in criterion[:2]:
                losses_l.append(function(pred_x, mask_x))
            loss_x = sum(losses_l) / 2

            loss_u_s1_arr = []
            for function in criterion[:2]:
                loss_u_s1_arr.append(function(pred_u_s1, pred_u_w.to(torch.float32)))
            loss_u_s1 = sum(loss_u_s1_arr) / 2
            loss_u_s1 = loss_u_s1 * ((conf_u_w_cutmixed1 >= config.semi.conf_thresh) & (ignore_mask_cutmixed1 != 255))
            loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()

            loss_u_s2_arr = []
            for function in criterion[:2]:
                loss_u_s2_arr.append(function(pred_u_s2, mask_u_w_cutmixed2[:, None, :, :].to(torch.float32)))
            loss_u_s2 = sum(loss_u_s2_arr) / 2
            loss_u_s2 = loss_u_s2 * ((conf_u_w_cutmixed2 >= config.semi.conf_thresh) & (ignore_mask_cutmixed2 != 255))
            loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()

            loss_u_w_fp_arr = []
            for function in criterion[:2]:
                loss_u_w_fp_arr.append(function(pred_u_w_fp, mask_u_w[:, None, :, :].to(torch.float32)))
            loss_u_w_fp = sum(loss_u_w_fp_arr) / 2
            loss_u_w_fp = loss_u_w_fp * ((conf_u_w >= config.semi.conf_thresh) & (ignore_mask != 255))
            loss_u_w_fp = loss_u_w_fp.sum() / (ignore_mask != 255).sum().item()


            loss_c_arr = []
            loss_c_s1 = criterion[-1](feat_u_w_cutmixed1, feat_u_s1)
            loss_c_arr.append(loss_c_s1)
            loss_c_s2 = criterion[-1](feat_u_w_cutmixed2, feat_u_s2)
            loss_c_arr.append(loss_c_s2)

            consistency_weight = get_current_consistency_weight(iter // 150)
            loss = loss_x + ((loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5) + 0.01 * (loss_c_s1 + loss_c_s2) / 2.0) * (sup_batch_len / unsup_batch_len) * consistency_weight


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            # Calculate and accumulate training loss
            loss_train_sum += loss.item() * sup_batch_len
            
            # Accumulate component losses
            bce_sup_loss_sum += losses_l[0].item() * sup_batch_len
            dice_sup_loss_sum += losses_l[1].item() * sup_batch_len
            bce_unsup_loss_sum += (loss_u_s1_arr[0].item() + loss_u_s2_arr[0].item() + loss_u_w_fp_arr[0].item()) * sup_batch_len
            dice_unsup_loss_sum += (loss_u_s1_arr[1].item() + loss_u_s2_arr[1].item() + loss_u_w_fp_arr[1].item()) * sup_batch_len
            corr_unsup_loss_sum += (loss_c_s1.item() + loss_c_s2.item()) * sup_batch_len


            # Calculate evaluation metrics
            with torch.no_grad():
                pred_x = pred_x.cpu().numpy() > 0.5
                mask_x = mask_x.cpu().numpy()
                assert (pred_x.shape == mask_x.shape), "Output and label shapes must match"
                
                # Calculate Dice and IoU metrics
                dice_train = metrics.dc(pred_x, mask_x)
                iou_train = metrics.jc(pred_x, mask_x)
                dice_train_sum += dice_train * sup_batch_len
                iou_train_sum += iou_train * sup_batch_len
                
                # Log training metrics
                metrics_str = (
                    f'Epoch {epoch}, iter {iter + 1}-'
                    f'Supervised Losses:-'
                    f'  - BCE Loss: {round(losses_l[0].item(), 5)}-'
                    f'  - Dice Loss: {round(losses_l[1].item(), 5)}-'
                    f'Unsupervised Losses (S1):-'
                    f'  - BCE Loss: {round(loss_u_s1_arr[0].item(), 5)}-'
                    f'  - Dice Loss: {round(loss_u_s1_arr[1].item(), 5)}-'
                    f'  - Correlation Loss: {round(loss_c_s1.item(), 5)}-'
                    f'Unsupervised Losses (S2):-'
                    f'  - BCE Loss: {round(loss_u_s2_arr[0].item(), 5)}-'
                    f'  - Dice Loss: {round(loss_u_s2_arr[1].item(), 5)}-'
                    f'  - Correlation Loss: {round(loss_c_s2.item(), 5)}-'
                    f'Unsupervised Feature Perturbation Losses:-'
                    f'  - BCE Loss: {round(loss_u_w_fp_arr[0].item(), 5)}-'
                    f'  - Dice Loss: {round(loss_u_w_fp_arr[1].item(), 5)}-'
                    f'Total Loss: {round(loss.item(), 5)}'
                )
                file_log.write(metrics_str + '-')
                file_log.flush()
                print(metrics_str)
            
            num_train += sup_batch_len
            iter += 1
            
            if config.debug:
                break

        # Calculate and log average training metrics
        avg_metrics_str = (
            f'Epoch {epoch}, Total train step {iter} || '
            f'AVG_loss: {round(loss_train_sum / num_train, 5)}, '
            f'Avg Dice score: {round(dice_train_sum/num_train, 4)}, '
            f'Avg IOU: {round(iou_train_sum/num_train, 4)}'
        )
        file_log.write(avg_metrics_str + '\n')
        file_log.flush()
        print(avg_metrics_str)
            
        # Calculate average component losses
        avg_bce_sup_loss = bce_sup_loss_sum / num_train
        avg_dice_sup_loss = dice_sup_loss_sum / num_train
        avg_bce_unsup_loss = bce_unsup_loss_sum / num_train
        avg_dice_unsup_loss = dice_unsup_loss_sum / num_train
        avg_corr_unsup_loss = corr_unsup_loss_sum / num_train

        # -----------------------------------------------------------------
        # Validation phase
        # -----------------------------------------------------------------
        model.eval()
        
        # Initialize validation metrics
        val_metrics = {
            'dice': [],
            'iou': [],
            'acc': [],
            'sen': [],
            'spe': [],
            'loss': []
        }
        num_val = 0
        
        with torch.no_grad():
            for batch in val_loader:
                img_x, mask_x = batch['image'].cuda(), batch['label'].cuda()
                batch_len = img_x.shape[0]
                
                # Forward pass
                pred = model(img_x)
                pred = torch.sigmoid(pred)
                
                # Calculate loss
                loss = criterion[0](pred, mask_x) + criterion[1](pred, mask_x)
                val_metrics['loss'].append(loss.item() * batch_len)
                
                # Calculate metrics
                pred_np = (pred > 0.5).cpu().numpy()
                mask_np = mask_x.cpu().numpy()
                metric = segmentation_metrics(pred_np, mask_np)
                for k, v in metric.items():
                    val_metrics[k].append(v * batch_len)
                
                num_val += batch_len
                
                if config.debug:
                    break
        
        # Calculate average validation metrics
        avg_metrics = {k: np.sum(v) / num_val for k, v in val_metrics.items()}
        
        # Check if current model is best
        is_best = avg_metrics['dice'] > best_metrics['val_dice']
        if is_best:
            best_metrics = {
                'val_dice': avg_metrics['dice'],
                'val_iou': avg_metrics['iou'],
                'val_acc': avg_metrics['acc'],
                'val_sen': avg_metrics['sen'],
                'val_spe': avg_metrics['spe']
            }
            max_dice = avg_metrics['dice']
            max_iou = avg_metrics['iou']
            best_epoch = epoch
        
        # Save checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, best_metrics, config, args, is_best)
        
        # Update learning rate
        scheduler.step()
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {loss_train_sum/num_train:.4f}")
        print(f"Val Metrics: {avg_metrics}")
        print(f"Best Metrics: {best_metrics}")
        print(f"Best Epoch: {best_epoch}")
        print("-" * 50)
        
        # Log to wandb
        wandb.log({
            # Training metrics
            "train/total_loss": loss_train_sum/num_train,
            "train/dice": dice_train_sum/num_train,
            "train/iou": iou_train_sum/num_train,
            
            # Component losses
            "train/supervised/bce_loss": avg_bce_sup_loss,
            "train/supervised/dice_loss": avg_dice_sup_loss,
            "train/unsupervised/bce_loss": avg_bce_unsup_loss,
            "train/unsupervised/dice_loss": avg_dice_unsup_loss,
            "train/unsupervised/corr_loss": avg_corr_unsup_loss,
            
            # Validation metrics
            "val/loss": avg_metrics["loss"],
            "val/dice": avg_metrics["dice"],
            "val/iou": avg_metrics["iou"],
            "val/acc": avg_metrics["acc"],
            "val/sensitivity": avg_metrics["sen"],
            "val/specificity": avg_metrics["spe"],
            
            # Other metrics
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epoch": epoch
        })

    # Log training completion
    completion_str = f'Complete training ---------------------------------------------------- \n The best epoch is {best_epoch}'
    file_log.write(completion_str + '\n')
    file_log.flush()
    print(completion_str)

    # Save checkpoints to wandb
    checkpoint_dir = f"checkpoints/{config.data.name}/{args.exp}_{config.data.supervised_ratio}/fold{args.fold}"
    artifact = wandb.Artifact(
        name=f"{args.exp}_{config.data.supervised_ratio}_fold{args.fold}",
        type="model",
        description=f"Model checkpoints for {args.exp} with {config.data.supervised_ratio} supervised ratio, fold {args.fold}"
    )
    artifact.add_dir(checkpoint_dir)
    wandb.log_artifact(artifact)
    print(f"Saved checkpoints to wandb artifacts from {checkpoint_dir}")
    wandb.finish()
    return




# ========================================================================================================
def test(config, model, model_dir, test_loader, criterion):
    # Load checkpoint
    checkpoint = torch.load(model_dir, map_location='cuda')
    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    dice_test_sum= 0
    iou_test_sum = 0
    loss_test_sum = 0
    num_test = 0
    for batch_id, batch in enumerate(test_loader):
        img = batch['image'].cuda().float()
        label = batch['label'].cuda().float()

        batch_len = img.shape[0]
            
        with torch.no_grad():
            output = model(img)
            output = torch.sigmoid(output)

            # calculate loss
            assert (output.shape == label.shape)
            losses = []
            for function in criterion[:2]:
                losses.append(function(output, label))
            loss_test_sum += sum(losses)*batch_len

            # calculate metrics
            output = output.cpu().numpy() > 0.5
            label = label.cpu().numpy()
            dice_test_sum += metrics.dc(output, label)*batch_len
            iou_test_sum += metrics.jc(output, label)*batch_len

            num_test += batch_len
            # end one test batch
            if config.debug: break

    # logging results for one dataset
    loss_test_epoch, dice_test_epoch, iou_test_epoch = loss_test_sum/num_test, dice_test_sum/num_test, iou_test_sum/num_test

    # logging average and store results
    with open(test_results_dir, 'w') as f:
        f.write(f'loss: {loss_test_epoch.item()}, Dice_score {dice_test_epoch}, IOU: {iou_test_epoch}')

    # print
    file_log.write('========================================================================================\n')
    file_log.write('Test || Average loss: {}, Dice score: {}, IOU: {}\n'.
                        format(round(loss_test_epoch.item(),5), 
                        round(dice_test_epoch,4), round(iou_test_epoch,4)))
    file_log.flush()
    print('========================================================================================')
    print('Test || Average loss: {}, Dice score: {}, IOU: {}'.
            format(round(loss_test_epoch.item(),5), 
            round(dice_test_epoch,4), round(iou_test_epoch,4)))

    wandb.log({
        "test/loss": loss_test_epoch.item(),
        "test/dice": dice_test_epoch,
        "test/iou": iou_test_epoch
    })

    return




if __name__=='__main__':
    now = datetime.now()
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description='Train experiment')
    parser.add_argument('--exp', type=str,default='tmp')
    parser.add_argument('--config_yml', type=str,default='Configs/multi_train_local.yml')
    parser.add_argument('--adapt_method', type=str, default=False)
    parser.add_argument('--num_domains', type=str, default=False)
    parser.add_argument('--dataset', type=str, nargs='+', default='isic2018')
    parser.add_argument('--k_fold', type=str, default='No')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
    parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    config = yaml.load(open(args.config_yml), Loader=yaml.FullLoader)
    config['model_adapt']['adapt_method']=args.adapt_method
    config['model_adapt']['num_domains']=args.num_domains
    config['data']['k_fold'] = args.k_fold
    config['seed'] = args.seed
    config['fold'] = args.fold
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    fix_all_seed(config['seed'])

    # print config and args
    print(yaml.dump(config, default_flow_style=False))
    for arg in vars(args):
        print("{:<20}: {}".format(arg, getattr(args, arg)))
    
    store_config = config
    config = DotDict(config)
    
    # logging tensorbord, config, best model
    exp_dir = '{}/{}_{}/fold{}'.format(config.data.save_folder, args.exp, config['data']['supervised_ratio'], args.fold)
    os.makedirs(exp_dir, exist_ok=True)
    best_model_dir = '{}/best.pth'.format(exp_dir)
    test_results_dir = '{}/test_results.txt'.format(exp_dir)

    # store yml file
    if config.debug == False:
        yaml.dump(store_config, open('{}/exp_config.yml'.format(exp_dir), 'w'))
    
    file_log = open('{}/log.txt'.format(exp_dir), 'w')
    main(config)
