'''
The default exp_name is tmp. Change it before formal training! isic2018 PH2 DMF SKD
nohup python -u multi_train_adapt.py --exp_name test --config_yml Configs/multi_train_local.yml --model MedFormer --batch_size 16 --adapt_method False --num_domains 1 --dataset PH2  --k_fold 4 > 4MedFormer_PH2.out 2>&1 &
'''
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

    
    model  = SwinUnet(img_size=config.data.img_size)




    total_trainable_params = sum(
                    p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print('{}M total parameters'.format(total_params/1e6))
    print('{}M total trainable parameters'.format(total_trainable_params/1e6))
    

    model = model.cuda()
    
    criterion = [nn.BCELoss(), dice_loss, corr_loss]

    # only test
    if config.test.only_test == True:
        test(config, model, config.test.test_model_dir, test_loader, criterion)
    else:
        train_val(config, model, train_loader, val_loader, criterion)
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
def train_val(config, model, train_loader, val_loader, criterion):
    # optimizer loss
    if config.train.optimizer.mode == 'adam':
        # optimizer = optim.Adam(model.parameters(), lr=float(config.train.optimizer.adam.lr))
        print('choose wrong optimizer')
    elif config.train.optimizer.mode == 'adamw':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),lr=float(config.train.optimizer.adamw.lr),
                                weight_decay=float(config.train.optimizer.adamw.weight_decay))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # ---------------------------------------------------------------------------
    # Training and Validating
    #----------------------------------------------------------------------------
    epochs = config.train.num_epochs
    max_iou = 0 # use for record best model
    max_dice = 0 # use for record best model
    best_epoch = 0 # use for recording the best epoch
    # create training data loading iteration
    
    torch.save(model.state_dict(), best_model_dir)
    for epoch in range(epochs):
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
                loss_u_s1_arr.append(function(pred_u_s1, mask_u_w_cutmixed1[:, None, :, :].to(torch.float32)))
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

            loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5) / 2.0 + 0.1 * (loss_c_s1 + loss_c_s2) / 2.0


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
                    f'Epoch {epoch}, iter {iter + 1}\n'
                    f'Supervised Losses:\n'
                    f'  - BCE Loss: {round(losses_l[0].item(), 5)}\n'
                    f'  - Dice Loss: {round(losses_l[1].item(), 5)}\n'
                    f'Unsupervised Losses (S1):\n'
                    f'  - BCE Loss: {round(loss_u_s1_arr[0].item(), 5)}\n'
                    f'  - Dice Loss: {round(loss_u_s1_arr[1].item(), 5)}\n'
                    f'  - Correlation Loss: {round(loss_c_s1.item(), 5)}\n'
                    f'Unsupervised Losses (S2):\n'
                    f'  - BCE Loss: {round(loss_u_s2_arr[0].item(), 5)}\n'
                    f'  - Dice Loss: {round(loss_u_s2_arr[1].item(), 5)}\n'
                    f'  - Correlation Loss: {round(loss_c_s2.item(), 5)}\n'
                    f'Unsupervised Feature Perturbation Losses:\n'
                    f'  - BCE Loss: {round(loss_u_w_fp_arr[0].item(), 5)}\n'
                    f'  - Dice Loss: {round(loss_u_w_fp_arr[1].item(), 5)}\n'
                    f'Total Loss: {round(loss.item(), 5)}'
                )
                file_log.write(metrics_str + '\n')
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
            'dice': 0, 'iou': 0, 'loss': 0, 'acc': 0,
            'sen': 0, 'spe': 0, 'pre': 0
        }
        num_val = 0

        for batch_id, batch in enumerate(val_loader):
            img = batch['image'].cuda().float()
            label = batch['label'].cuda().float()
            batch_len = img.shape[0]

            with torch.no_grad():
                # Forward pass
                output = model(img)
                output = torch.sigmoid(output)

                # Calculate validation loss
                assert (output.shape == label.shape), "Output and label shapes must match"
                losses = [function(output, label) for function in criterion[:2]]
                val_metrics['loss'] += sum(losses) * batch_len / 2

                # Calculate validation metrics
                output_np = output.cpu().numpy() > 0.5
                label_np = label.cpu().numpy()
                val_metrics['dice'] += metrics.dc(output_np, label_np) * batch_len
                val_metrics['iou'] += metrics.jc(output_np, label_np) * batch_len

                # Calculate segmentation metrics
                output_bin = (output > 0.5).float().cpu()
                label_bin = label.float().cpu()
                seg_metrics = segmentation_metrics(output_bin, label_bin)
                
                for metric in ['acc', 'sen', 'spe', 'pre']:
                    val_metrics[metric] += seg_metrics[metric.upper()] * batch_len

                num_val += batch_len
                
                if config.debug:
                    break

        # Calculate average validation metrics
        for metric in val_metrics:
            val_metrics[metric] /= num_val

        # Log validation results
        val_str = (
            f'Epoch {epoch}, Validation || '
            f'sum_loss: {round(val_metrics["loss"].item(), 5)}, '
            f'Dice score: {round(val_metrics["dice"], 4)}, '
            f'IOU: {round(val_metrics["iou"], 4)}'
        )
        file_log.write(val_str + '\n')
        file_log.flush()
        print(val_str)

        # Update learning rate
        scheduler.step()

        # Save best model
        if val_metrics['dice'] > max_dice:
            torch.save(model.state_dict(), best_model_dir)
            max_dice = val_metrics['dice']
            best_epoch = epoch
            best_str = f'New best epoch {epoch}!==============================='
            file_log.write(best_str + '\n')
            file_log.flush()
            print(best_str)
        
        # Log training time
        end = time.time()
        time_elapsed = end - start
        time_str = (
            f'Training and evaluating on epoch{epoch} complete in '
            f'{time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s'
        )
        file_log.write(time_str + '\n')
        file_log.flush()
        print(time_str)

        if config.debug:
            return

        # Log metrics to wandb
        wandb.log({
            # Training metrics
            "train/loss": loss_train_sum / num_train,
            "train/dice": dice_train_sum / num_train,
            "train/iou": iou_train_sum / num_train,
            
            # Supervised losses
            "train/bce_sup_loss": avg_bce_sup_loss,
            "train/dice_sup_loss": avg_dice_sup_loss,
            
            # Unsupervised losses
            "train/bce_unsup_loss": avg_bce_unsup_loss,
            "train/dice_unsup_loss": avg_dice_unsup_loss,
            "train/corr_unsup_loss": avg_corr_unsup_loss,
            
            # Validation metrics
            "val/loss": val_metrics["loss"].item(),
            "val/dice": val_metrics["dice"],
            "val/iou": val_metrics["iou"],
            "val/acc": val_metrics["acc"],
            "val/sensitivity": val_metrics["sen"],
            "val/specificity": val_metrics["spe"],
            "val/precision": val_metrics["pre"],
            
            # Other metrics
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0]
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

    return




# ========================================================================================================
def test(config, model, model_dir, test_loader, criterion):
    model.load_state_dict(torch.load(model_dir))
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
