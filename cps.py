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
import torch.utils.data
import torch.optim as optim
import medpy.metric.binary as metrics

from Datasets.create_dataset import *
from Datasets.transform import normalize
from Utils.losses import dice_loss
from Utils.pieces import DotDict
from Utils.functions import fix_all_seed, segmentation_metrics

from Models.Transformer.SwinUnet import SwinUnet
from itertools import cycle

torch.cuda.empty_cache()

def main(config):
    
    wandb.init(project="SkinSeg", name=f"CPS_2_fold{config.fold}", config=config)
    
    dataset = get_dataset(config, img_size=config.data.img_size, 
                                                    supervised_ratio=config.data.supervised_ratio, 
                                                    train_aug=config.data.train_aug,
                                                    k=config.fold,
                                                    ulb_dataset=StrongWeakAugment2,
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

    
    model1  = SwinUnet(img_size=config.data.img_size)
    model2  = SwinUnet(img_size=config.data.img_size)



    total_trainable_params = sum(
                    p.numel() for p in model1.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model1.parameters())
    print('{}M total parameters'.format(total_params/1e6))
    print('{}M total trainable parameters'.format(total_trainable_params/1e6))
    
    # from thop import profile
    # input = torch.randn(1,3,224,224)
    # flops, params = profile(model, (input,))
    # print(f"total flops : {flops/1e9} G")

    # test model
    # x = torch.randn(5,3,224,224)
    # y = model(x)
    # print(y.shape)

    model1 = model1.cuda()
    model2 = model2.cuda()
    
    criterion = [nn.BCELoss(), dice_loss]

    model = train_val(config, model1, model2, train_loader, val_loader, criterion)
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
def train_val(config, model1, model2, train_loader, val_loader, criterion):
    # optimizer loss
    if config.train.optimizer.mode == 'adam':
        # optimizer = optim.Adam(model.parameters(), lr=float(config.train.optimizer.adam.lr))
        print('choose wrong optimizer')
    elif config.train.optimizer.mode == 'adamw':
        optimizer1 = optim.AdamW(filter(lambda p: p.requires_grad, model1.parameters()),lr=float(config.train.optimizer.adamw.lr),
                                weight_decay=float(config.train.optimizer.adamw.weight_decay))
        optimizer2 = optim.AdamW(filter(lambda p: p.requires_grad, model2.parameters()),lr=float(config.train.optimizer.adamw.lr),
                                weight_decay=float(config.train.optimizer.adamw.weight_decay))
    scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=50, gamma=0.5)
    scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=50, gamma=0.5)

    # ---------------------------------------------------------------------------
    # Training and Validating
    #----------------------------------------------------------------------------
    epochs = config.train.num_epochs
    max_iou = 0 # use for record best model
    max_dice = 0 # use for record best model
    best_epoch = 0 # use for recording the best epoch
    model = model1
    # create training data loading iteration
    
    torch.save(model.state_dict(), best_model_dir)
    for epoch in range(epochs):
        start = time.time()
        # ----------------------------------------------------------------------
        # train
        # ---------------------------------------------------------------------
        model1.train()
        model2.train()
        dice_train_sum_1 = 0
        iou_train_sum_1 = 0
        dice_train_sum_2 = 0
        iou_train_sum_2 = 0
        loss_train_sum = 0
        num_train = 0
        iter = 0
        bce_sup_loss_sum_1 = 0
        dice_sup_loss_sum_1 = 0
        bce_unsup_loss_sum_1 = 0
        dice_unsup_loss_sum_1 = 0
        bce_sup_loss_sum_2 = 0
        dice_sup_loss_sum_2 = 0
        bce_unsup_loss_sum_2 = 0
        dice_unsup_loss_sum_2 = 0
        source_dataset = zip(cycle(train_loader['l_loader']), train_loader['u_loader'])
        for idx, (batch, batch_w_s) in enumerate(source_dataset):
            img = batch['image'].cuda().float()
            label = batch['label'].cuda().float()
            weak_batch = batch_w_s['img_w'].cuda().float()
            
            sup_batch_len = img.shape[0]
            unsup_batch_len = weak_batch.shape[0]
            
            output1 = model1(img)
            output1 = torch.sigmoid(output1)
            
            # calculate loss
            losses_l1 = []
            for function in criterion:
                losses_l1.append(function(output1, label))
                
            output2 = model2(img)
            output2 = torch.sigmoid(output2)
            
            losses_l2 = []
            for function in criterion:
                losses_l2.append(function(output2, label))
            
            # FixMatch
            #======================================================================================================
            # outputs for model
            outputs_u1 = model1(weak_batch)
            outputs_u1 = torch.sigmoid(outputs_u1)
            pseudo_u1 = torch.round(outputs_u1)
            
            outputs_u2 = model2(weak_batch)
            outputs_u2 = torch.sigmoid(outputs_u2)
            pseudo_u2 = torch.round(outputs_u2)
            
            # calculate loss
            losses_u1 = []
            for function in criterion:
                losses_u1.append(function(outputs_u1, pseudo_u2))
                
            losses_u2 = []
            for function in criterion:
                losses_u2.append(function(outputs_u2, pseudo_u1))
            #======================================================================================================
            consistency_weight = get_current_consistency_weight(iter // 150)
            
            sup_loss_1 = (losses_l1[0] + losses_l1[1]) / 2.0
            unsup_loss_1 = (losses_u1[0] + losses_u1[1]) / 2.0
            loss_1 = sup_loss_1 + unsup_loss_1 * consistency_weight * (sup_batch_len / unsup_batch_len)
            
            sup_loss_2 = (losses_l2[0] + losses_l2[1]) / 2.0
            unsup_loss_2 = (losses_u2[0] + losses_u2[1]) / 2.0
            loss_2 = sup_loss_2 + unsup_loss_2 * consistency_weight * (sup_batch_len / unsup_batch_len)
            
            loss = loss_1 + loss_2
            
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            
            loss.backward()
            
            optimizer1.step()
            optimizer2.step()
            
            
            loss_train_sum += loss.item() * sup_batch_len
            
            # calculate metrics
            with torch.no_grad():
                output1 = output1.cpu().numpy() > 0.5
                output2 = output2.cpu().numpy() > 0.5
                label = label.cpu().numpy()

                dice_train_1 = metrics.dc(output1, label)
                iou_train_1 = metrics.jc(output1, label)
                dice_train_sum_1 += dice_train_1 * sup_batch_len
                iou_train_sum_1 += iou_train_1 * sup_batch_len
                
                dice_train_2 = metrics.dc(output2, label)
                iou_train_2 = metrics.jc(output2, label)
                dice_train_sum_2 += dice_train_2 * sup_batch_len
                iou_train_sum_2 += iou_train_2 * sup_batch_len
                
            # Cộng dồn loss thành phần cho model 1
            bce_sup_loss_sum_1 += losses_l1[0].item() * sup_batch_len
            dice_sup_loss_sum_1 += losses_l1[1].item() * sup_batch_len
            bce_unsup_loss_sum_1 += losses_u1[0].item() * sup_batch_len
            dice_unsup_loss_sum_1 += losses_u1[1].item() * sup_batch_len
            
            # Cộng dồn loss thành phần cho model 2
            bce_sup_loss_sum_2 += losses_l2[0].item() * sup_batch_len
            dice_sup_loss_sum_2 += losses_l2[1].item() * sup_batch_len
            bce_unsup_loss_sum_2 += losses_u2[0].item() * sup_batch_len
            dice_unsup_loss_sum_2 += losses_u2[1].item() * sup_batch_len
            
            file_log.write('Epoch {}, iter {}:\n'.format(epoch, iter + 1))
            file_log.write('Dice Sup Loss 1: {}, Dice Unsup Loss 1: {}, BCE Sup Loss 1: {}, BCE UnSup Loss 1: {}\n'.format(
                round(losses_l1[1].item(), 5), round(losses_u1[1].item(), 5), round(losses_l1[0].item(), 5), round(losses_u1[0].item(), 5)
            ))
            file_log.write('Dice Sup Loss 2: {}, Dice Unsup Loss 2: {}, BCE Sup Loss 2: {}, BCE UnSup Loss 2: {}\n'.format(
                round(losses_l2[1].item(), 5), round(losses_u2[1].item(), 5), round(losses_l2[0].item(), 5), round(losses_u2[0].item(), 5)
            ))
            file_log.flush()
            
            # print('Epoch {}, iter {}:'.format(epoch, iter + 1))
            # print('Dice Sup Loss 1: {}, Dice Unsup Loss 1: {}, BCE Sup Loss 1: {}, BCE UnSup Loss 1: {}'.format(
            #     round(losses_l1[1].item(), 5), round(losses_u1[1].item(), 5), round(losses_l1[0].item(), 5), round(losses_u1[0].item(), 5)
            # ))
            # print('Dice Sup Loss 2: {}, Dice Unsup Loss 2: {}, BCE Sup Loss 2: {}, BCE UnSup Loss 2: {}'.format(
            #     round(losses_l2[1].item(), 5), round(losses_u2[1].item(), 5), round(losses_l2[0].item(), 5), round(losses_u2[0].item(), 5)
            # ))
            
            num_train += sup_batch_len
            iter += 1
            
            # end one test batch
            if config.debug: break
                

        # print
        file_log.write('Epoch {}, Total train step {} || AVG_loss: {}, Avg Dice score 1: {}, Avg IOU 1: {}, Avg Dice score 2: {}, Avg IOU 2: {}\n'.format(
            epoch, 
            iter, 
            round(loss_train_sum / num_train,5), 
            round(dice_train_sum_1/num_train,4), 
            round(iou_train_sum_1/num_train,4),
            round(dice_train_sum_2/num_train,4), 
            round(iou_train_sum_2/num_train,4)))
        file_log.flush()
        print('Epoch {}, Total train step {} || AVG_loss: {}, Avg Dice score 1: {}, Avg IOU 1: {}, Avg Dice score 2: {}, Avg IOU 2: {}'.format(
            epoch, 
            iter, 
            round(loss_train_sum / num_train,5), 
            round(dice_train_sum_1/num_train,4), 
            round(iou_train_sum_1/num_train,4),
            round(dice_train_sum_2/num_train,4), 
            round(iou_train_sum_2/num_train,4)))
            
        
        # -----------------------------------------------------------------
        # validate 1
        # ----------------------------------------------------------------
        model1.eval()
        
        dice_val_sum= 0
        iou_val_sum = 0
        loss_val_sum = 0
        num_val = 0
        acc_val_sum = 0
        sen_val_sum = 0
        spe_val_sum = 0
        pre_val_sum = 0

        for batch_id, batch in enumerate(val_loader):
            img = batch['image'].cuda().float()
            label = batch['label'].cuda().float()
            
            batch_len = img.shape[0]

            with torch.no_grad():
                output = model1(img)
                    
                output = torch.sigmoid(output)

                # calculate loss
                assert (output.shape == label.shape)
                losses = []
                for function in criterion:
                    losses.append(function(output, label))
                loss_val_sum += sum(losses)*batch_len / 2

                # calculate metrics
                output_np = output.cpu().numpy() > 0.5
                label_np = label.cpu().numpy()
                dice_val_sum += metrics.dc(output_np, label_np)*batch_len
                iou_val_sum += metrics.jc(output_np, label_np)*batch_len

                # Tính các metrics segmentation cho val (dùng tensor float)
                output_bin = (output > 0.5).float().cpu()
                label_bin = label.float().cpu()
                seg_metrics = segmentation_metrics(output_bin, label_bin)
                acc_val_sum += seg_metrics['ACC'] * batch_len
                sen_val_sum += seg_metrics['SEN'] * batch_len
                spe_val_sum += seg_metrics['SPE'] * batch_len
                pre_val_sum += seg_metrics['PRE'] * batch_len

                num_val += batch_len
                # end one val batch
                if config.debug: break

        # logging per epoch for one dataset
        loss_val_epoch, dice_val_epoch, iou_val_epoch = loss_val_sum/num_val, dice_val_sum/num_val, iou_val_sum/num_val
        acc_val_epoch = acc_val_sum / num_val
        sen_val_epoch = sen_val_sum / num_val
        spe_val_epoch = spe_val_sum / num_val
        pre_val_epoch = pre_val_sum / num_val
        
        # print
        file_log.write('Epoch {}, Model 1, Validation || sum_loss: {}, Dice score: {}, IOU: {}\n'.
                format(epoch, round(loss_val_epoch.item(),5), 
                round(dice_val_epoch,4), round(iou_val_epoch,4)))
        file_log.flush()
        print('Epoch {}, Model 1, Validation || sum_loss: {}, Dice score: {}, IOU: {}'.
                format(epoch, round(loss_val_epoch.item(),5), 
                round(dice_val_epoch,4), round(iou_val_epoch,4)))
        
        avg_dice_1 = dice_val_epoch
        avg_iou_1 = iou_val_epoch
        
        # -----------------------------------------------------------------
        # validate 2
        # ----------------------------------------------------------------
        model2.eval()
        
        dice_val_sum= 0
        iou_val_sum = 0
        loss_val_sum = 0
        num_val = 0
        acc_val_sum = 0
        sen_val_sum = 0
        spe_val_sum = 0
        pre_val_sum = 0

        for batch_id, batch in enumerate(val_loader):
            img = batch['image'].cuda().float()
            label = batch['label'].cuda().float()
            
            batch_len = img.shape[0]

            with torch.no_grad():
                output = model2(img)
                    
                output = torch.sigmoid(output)

                # calculate loss
                assert (output.shape == label.shape)
                losses = []
                for function in criterion:
                    losses.append(function(output, label))
                loss_val_sum += sum(losses)*batch_len / 2

                # calculate metrics
                output_np = output.cpu().numpy() > 0.5
                label_np = label.cpu().numpy()
                dice_val_sum += metrics.dc(output_np, label_np)*batch_len
                iou_val_sum += metrics.jc(output_np, label_np)*batch_len

                # Tính các metrics segmentation cho val (dùng tensor float)
                output_bin = (output > 0.5).float().cpu()
                label_bin = label.float().cpu()
                seg_metrics = segmentation_metrics(output_bin, label_bin)
                acc_val_sum += seg_metrics['ACC'] * batch_len
                sen_val_sum += seg_metrics['SEN'] * batch_len
                spe_val_sum += seg_metrics['SPE'] * batch_len
                pre_val_sum += seg_metrics['PRE'] * batch_len

                num_val += batch_len
                # end one val batch
                if config.debug: break

        # logging per epoch for one dataset
        loss_val_epoch, dice_val_epoch, iou_val_epoch = loss_val_sum/num_val, dice_val_sum/num_val, iou_val_sum/num_val
        acc_val_epoch = acc_val_sum / num_val
        sen_val_epoch = sen_val_sum / num_val
        spe_val_epoch = spe_val_sum / num_val
        pre_val_epoch = pre_val_sum / num_val
        
        # print
        file_log.write('Epoch {}, Model 2, Validation || sum_loss: {}, Dice score: {}, IOU: {}\n'.
                format(epoch, round(loss_val_epoch.item(),5), 
                round(dice_val_epoch,4), round(iou_val_epoch,4)))
        file_log.flush()
        print('Epoch {}, Model 2, Validation || sum_loss: {}, Dice score: {}, IOU: {}'.
                format(epoch, round(loss_val_epoch.item(),5), 
                round(dice_val_epoch,4), round(iou_val_epoch,4)))
        
        avg_dice_2 = dice_val_epoch
        avg_iou_2 = iou_val_epoch


        # scheduler step, record lr
        scheduler1.step()
        scheduler2.step()

        # store model using the average iou
        if avg_dice_1 > avg_dice_2:
            avg_dice_val_epoch = avg_dice_1
            model = model1
        else:
            avg_dice_val_epoch = avg_dice_2
            model = model2
        if avg_dice_val_epoch > max_dice:
            torch.save(model.state_dict(), best_model_dir)
            max_dice = avg_dice_val_epoch
            best_epoch = epoch
            file_log.write('New best epoch {}!===============================\n'.format(epoch))
            file_log.flush()
            print('New best epoch {}!==============================='.format(epoch))
        
        end = time.time()
        time_elapsed = end-start
        file_log.write('Training and evaluating on epoch{} complete in {:.0f}m {:.0f}s\n'.
                    format(epoch, time_elapsed // 60, time_elapsed % 60))
        file_log.flush()
        print('Training and evaluating on epoch{} complete in {:.0f}m {:.0f}s'.
            format(epoch, time_elapsed // 60, time_elapsed % 60))

        # end one epoch
        if config.debug: return

        wandb.log({
            "train/loss": loss_train_sum / num_train,
            "train/dice_1": dice_train_sum_1 / num_train,
            "train/iou_1": iou_train_sum_1 / num_train,
            "train/dice_2": dice_train_sum_2 / num_train,
            "train/iou_2": iou_train_sum_2 / num_train,
            "train/bce_sup_loss_1": bce_sup_loss_sum_1 / num_train,
            "train/dice_sup_loss_1": dice_sup_loss_sum_1 / num_train,
            "train/bce_unsup_loss_1": bce_unsup_loss_sum_1 / num_train,
            "train/dice_unsup_loss_1": dice_unsup_loss_sum_1 / num_train,
            "train/bce_sup_loss_2": bce_sup_loss_sum_2 / num_train,
            "train/dice_sup_loss_2": dice_sup_loss_sum_2 / num_train,
            "train/bce_unsup_loss_2": bce_unsup_loss_sum_2 / num_train,
            "train/dice_unsup_loss_2": dice_unsup_loss_sum_2 / num_train,
            "val/loss_1": loss_val_epoch.item(),
            "val/dice_1": avg_dice_1,
            "val/iou_1": avg_iou_1,
            "val/acc_1": acc_val_epoch,
            "val/sensitivity_1": sen_val_epoch,
            "val/specificity_1": spe_val_epoch,
            "val/precision_1": pre_val_epoch,
            "val/loss_2": loss_val_epoch.item(),
            "val/dice_2": avg_dice_2,
            "val/iou_2": avg_iou_2,
            "val/acc_2": acc_val_epoch,
            "val/sensitivity_2": sen_val_epoch,
            "val/specificity_2": spe_val_epoch,
            "val/precision_2": pre_val_epoch,
            "epoch": epoch,
            "lr_1": scheduler1.get_last_lr()[0],
            "lr_2": scheduler2.get_last_lr()[0]
        })
    
    file_log.write('Complete training ---------------------------------------------------- \n The best epoch is {}\n'.format(best_epoch))
    file_log.flush()
    
    print('Complete training ---------------------------------------------------- \n The best epoch is {}'.format(best_epoch))

    # Lưu folder checkpoints lên wandb sử dụng Artifacts
    checkpoint_dir = f"checkpoints/{config.data.name}/{args.exp}_{config.data.supervised_ratio}/fold{args.fold}"
    artifact = wandb.Artifact(
        name=f"{args.exp}_{config.data.supervised_ratio}_fold{args.fold}",
        type="model",
        description=f"Model checkpoints for {args.exp} with {config.data.supervised_ratio} supervised ratio, fold {args.fold}"
    )
    artifact.add_dir(checkpoint_dir)
    wandb.log_artifact(artifact)
    print(f"Saved checkpoints to wandb artifacts from {checkpoint_dir}")

    return model




# ========================================================================================================
def test(config, model, model_dir, test_loader, criterion):
    model.load_state_dict(torch.load(model_dir))
    model.eval()
    dice_test_sum= 0
    iou_test_sum = 0
    loss_test_sum = 0
    num_test = 0
    acc_test_sum = 0
    sen_test_sum = 0
    spe_test_sum = 0
    pre_test_sum = 0
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
            for function in criterion:
                losses.append(function(output, label))
            loss_test_sum += sum(losses)*batch_len

            # calculate metrics
            output_np = output.cpu().numpy() > 0.5
            label_np = label.cpu().numpy()
            dice_test_sum += metrics.dc(output_np, label_np)*batch_len
            iou_test_sum += metrics.jc(output_np, label_np)*batch_len

            # Tính các metrics segmentation cho test (dùng tensor float)
            output_bin = (output > 0.5).float().cpu()
            label_bin = label.float().cpu()
            seg_metrics = segmentation_metrics(output_bin, label_bin)
            acc_test_sum += seg_metrics['ACC'] * batch_len
            sen_test_sum += seg_metrics['SEN'] * batch_len
            spe_test_sum += seg_metrics['SPE'] * batch_len
            pre_test_sum += seg_metrics['PRE'] * batch_len

            num_test += batch_len
            # end one test batch
            if config.debug: break

    # logging results for one dataset
    loss_test_epoch, dice_test_epoch, iou_test_epoch = loss_test_sum/num_test, dice_test_sum/num_test, iou_test_sum/num_test
    acc_test_epoch = acc_test_sum / num_test
    sen_test_epoch = sen_test_sum / num_test
    spe_test_epoch = spe_test_sum / num_test
    pre_test_epoch = pre_test_sum / num_test

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
        "test/iou": iou_test_epoch,
        "test/acc": acc_test_epoch,
        "test/sensitivity": sen_test_epoch,
        "test/specificity": spe_test_epoch,
        "test/precision": pre_test_epoch
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
