import warnings 
warnings.filterwarnings('ignore')
import numpy as np
import torch.nn as nn
import torch
import argparse
import random
import os
import time
from torchvision.transforms import Compose, ToTensor, RandomCrop
from metrics.miou import mIOUMetrics
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from dataset.densepass_val_dataset import densepass_val
from dataset.city.City_dataset import CityDataset
from dataset.densepass_train_dataset import densepass_train
from dataset.equi2tangent import eq2tangent
from models.discriminator import FCDiscriminator
from models.segformer.segformer import Seg
from info_nce import InfoNCE

tangent_batch = 18

NAME_CLASSES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "light",
    "sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motocycle",
    "bicycle"]

class ToLabel:

    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long()
        
        #return np.asarray(image, np.float32)

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(epoch,max_epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return sigmoid_rampup(epoch, max_epoch)

def batch_erp2tangent(batch_erp, tangent_size):
    bs = batch_erp.size(0)
    dim = batch_erp.size(1)
    batch_tangent = []
    for i in range(bs):
        erp = batch_erp[i]
        tangent = torch.tensor(eq2tangent(erp.permute(1,2,0), height=tangent_size, width=tangent_size)).reshape(tangent_size,tangent_size,18,dim).permute(2,3,0,1) 
        batch_tangent.append(tangent) 
    batch_tangent= torch.tensor([item.cpu().detach().numpy() for item in batch_tangent]) 
    return batch_tangent.reshape(tangent_batch,dim,tangent_size,tangent_size)

def CityCrop(s_img,s_gt,tangent_size,it):
    trans = RandomCrop(tangent_size)
    pseudo_tangent = []
    pseudo_tangent_label = []
    seed = it
    for i in range(18):
        torch.random.manual_seed(seed)
        tangent_ = trans(s_img) 
        torch.random.manual_seed(seed)
        tangent_label = trans(s_gt)
        pseudo_tangent.append(tangent_)
        pseudo_tangent_label.append(tangent_label)
    pseudo_tangent= torch.tensor([item.cpu().detach().numpy() for item in pseudo_tangent]).cuda().squeeze(1) # [18, 3, tangent_size, tangent_size]
    pseudo_tangent_label= torch.tensor([item.cpu().detach().numpy() for item in pseudo_tangent_label]).cuda().squeeze(1) # [18, tangent_size, tangent_size]
    return pseudo_tangent.reshape(tangent_batch,3,tangent_size,tangent_size), pseudo_tangent_label.reshape(tangent_batch,tangent_size,tangent_size)

def adjust_learning_rate_poly(optimizer, epoch, num_epochs, base_lr, power):
    lr = base_lr * (1-epoch/num_epochs)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def main():
    parser = argparse.ArgumentParser(description='pytorch implemention')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 6)')
    parser.add_argument('--iterations', type=int, default=30000, metavar='N',
                        help='number of epochs to train (default: 30000)')
    parser.add_argument('--lr', type=float, default=6e-5, metavar='LR',
                        help='learning rate (default: 6e-5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--save_root', default = '',
                        help='Please add your model save directory') 
    parser.add_argument('--exp_name', default = '',
                        help='')
    parser.add_argument('--sup_set', type=str, default='train', help='supervised training set')
    parser.add_argument('--cutmix', default =False, help='cutmix')
    #================================hyper parameters================================#
    parser.add_argument('--alpha', type=float, default =0.5, help='alpha')
    parser.add_argument('--lamda', type=float, default =0.001, help='lamda')
    #================================================================================#
    args = parser.parse_args()
    best_performance = 0.0

    save_path = "{}{}".format(args.save_root,args.exp_name)
    writer = SummaryWriter(log_dir=save_path)

    if os.path.exists(save_path):
        pass
    else:
        os.makedirs(save_path)   

    torch.cuda.set_device(args.local_rank)
    with torch.cuda.device(args.local_rank):
        dist.init_process_group(backend='nccl',init_method='env://') #nccl
        if dist.get_rank() == 0:
            print(args)
            print('init cnn lr: {}, batch size: {}, gpus:{}'.format(args.lr, args.batch_size, dist.get_world_size()))

        num_classes = 19
        # Cityscapes dataset
        # ------------------------------------------------------------------------------------------------------------#
        img_mean=[0.485, 0.456, 0.406]
        img_std=[0.229, 0.224, 0.225]
        city_crop_size = 512
        city_dataset_path = "./" # cityscapes dataset root
        city_label_dataset = CityDataset(f'{city_dataset_path}',split='train', base_size=2048, crop_size=city_crop_size, norm_mean=img_mean, norm_std=img_std)        
        city_label_sampler = DistributedSampler(city_label_dataset, num_replicas=dist.get_world_size()) 
        city_label_loader = torch.utils.data.DataLoader(city_label_dataset,batch_size=args.batch_size,sampler=city_label_sampler,num_workers=12,worker_init_fn=lambda x: random.seed(time.time() + x),drop_last=True,)
        
        # city_val_dataset = CityDataset(f'{city_dataset_path}', split='val', base_size=2048, crop_size=city_crop_size, norm_mean=img_mean, norm_std=img_std)
        # val_loader = torch.utils.data.DataLoader(city_val_dataset,batch_size=args.batch_size,shuffle=False,num_workers=12)
        # DensePASS dataset
        # ------------------------------------------------------------------------------------------------------------#
        input_transform_cityscapes = Compose([ToTensor(),])
        target_transform_cityscapes = Compose([ToLabel(),])  
        train_root = "./" # training set root
        val_root = "./"   # validation set root
        train_DensePASS = densepass_train(train_root, list_path='./train.txt',set=None)
        val_DensePASS = densepass_val(val_root, input_transform=input_transform_cityscapes,target_transform=target_transform_cityscapes, target=True)
        pass_train_sampler = DistributedSampler(train_DensePASS, num_replicas=dist.get_world_size())
        pass_train_loader = torch.utils.data.DataLoader(train_DensePASS,batch_size=args.batch_size,sampler=pass_train_sampler,num_workers=12,worker_init_fn=lambda x: random.seed(time.time() + x),drop_last=True,)
        pass_val_loader = torch.utils.data.DataLoader(val_DensePASS,batch_size=args.batch_size,shuffle=False,num_workers=12)
        # Models
        # ------------------------------------------------------------------------------------------------------------#
        tangent_size = 224
        model1 = Seg(backbone='mit_b1',num_classes=num_classes,embedding_dim=512,pretrained=True,height=400,width=2048)
        model2 = Seg(backbone='mit_b1',num_classes=num_classes,embedding_dim=512,pretrained=True,height=224,width=224)
        model_path = "model.pth" # source domain pre-trained model 
        model1.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")),strict=False)
        model2.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")),strict=False)
        model1 = model1.to(args.local_rank)
        model1 = DistributedDataParallel(model1,device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False, find_unused_parameters=True)
        model2 = model2.to(args.local_rank)
        model2 = DistributedDataParallel(model2,device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False, find_unused_parameters=True)
        
        modelD1 = FCDiscriminator(num_classes=512).to(args.local_rank)
        modelD2 = FCDiscriminator(num_classes=512).to(args.local_rank)
        d_lr = 0.0000001
        optimizerD1 = optim.Adam(modelD1.parameters(), lr=d_lr, betas=(0.9, 0.99))
        optimizerD2 = optim.Adam(modelD2.parameters(), lr=d_lr, betas=(0.9, 0.99))
        # Training Details
        # ------------------------------------------------------------------------------------------------------------#
        epoch = 0
        kl_loss = nn.KLDivLoss(size_average=None, reduce=None, reduction='mean', log_target=True)
        bce_loss = torch.nn.BCEWithLogitsLoss()
        # Iterative dataloader
        # ------------------------------------------------------------------------------------------------------------#        
        city_sup_loader = iter(city_label_loader)
        pass_img_loader = iter(pass_train_loader)
        city_length = len(city_sup_loader)
        pass_length = len(pass_img_loader)
        print(f'Panoramic Dataset length:{len(train_DensePASS)};')
        print(f'Pinhole Dataset length:{len(city_label_dataset)};')
        # Training Details
        # ------------------------------------------------------------------------------------------------------------#
        weight = torch.ones(num_classes)
        weight[0] = 2.8149201869965
        weight[1] = 6.9850029945374
        weight[2] = 3.7890393733978
        weight[3] = 9.9428062438965
        weight[4] = 9.7702074050903
        weight[5] = 9.5110931396484
        weight[6] = 10.311357498169
        weight[7] = 10.026463508606
        weight[8] = 4.6323022842407
        weight[9] = 9.5608062744141
        weight[10] = 7.8698215484619
        weight[11] = 9.5168733596802
        weight[12] = 10.373730659485
        weight[13] = 6.6616044044495
        weight[14] = 10.260489463806
        weight[15] = 10.287888526917
        weight[16] = 10.289801597595
        weight[17] = 10.405355453491
        weight[18] = 10.138095855713
        weight = weight.to(args.local_rank)
        criterion_sup = nn.CrossEntropyLoss(weight=weight, reduction='mean', ignore_index=255)
        optimizer1 = optim.AdamW(model1.parameters(), lr=args.lr, weight_decay=0.0001)
        optimizer2 = optim.AdamW(model2.parameters(), lr=args.lr, weight_decay=0.0001)
        # Iterative dataloader
        # ------------------------------------------------------------------------------------------------------------#        
        city_sup_loader = iter(city_label_loader)
        pass_img_loader = iter(pass_train_loader)
        city_length = len(city_sup_loader)
        pass_length = len(pass_img_loader)
        max_epoch = args.iterations / pass_length
        print(max_epoch)
        print(f'Panoramic Dataset length:{len(train_DensePASS)};')
        print(f'Pinhole Dataset length:{len(city_label_dataset)};')
        # Training Iterations
        # ------------------------------------------------------------------------------------------------------------#        
        for it in range(1, args.iterations + 1):
            since = time.time()
            if it % city_length == 0:
                city_label_loader.sampler.set_epoch(epoch)
                city_sup_loader = iter(city_label_loader)
            if it % pass_length == 0:
                pass_train_loader.sampler.set_epoch(epoch)
                pass_img_loader = iter(pass_train_loader)
            s_img, s_gt = city_sup_loader.__next__()
            s_img, s_gt = s_img.to(args.local_rank), s_gt.to(args.local_rank)
            p_img, _, _ = pass_img_loader.__next__()
            p_img = p_img.to(args.local_rank) 
            # Image Process
            # ------------------------------------------------------------------------------------------------------------#        
            tangent = batch_erp2tangent(p_img,tangent_size)
            tangent = tangent.to(args.local_rank) 
            pseudo_tangent, pseudo_tangent_label = CityCrop(s_img,s_gt,tangent_size,it)
            pseudo_tangent, pseudo_tangent_label = pseudo_tangent.to(args.local_rank), pseudo_tangent_label.to(args.local_rank)
            # Model1 Prediction
            # ------------------------------------------------------------------------------------------------------------#        
            city_pred, city_feat = model1(s_img) 
            erp_pred, erp_feat = model1(p_img) 
            tangent_proj = batch_erp2tangent(erp_pred,tangent_size)
            tangent_proj = tangent_proj.to(args.local_rank)
            # Model2 Prediction
            # ------------------------------------------------------------------------------------------------------------#        
            tangent_pred, tangent_feat = model2(tangent) 
            pseudo_tangent_pred, pseudo_tangent_feat = model2(pseudo_tangent) 
            # Loss calculation
            # ------------------------------------------------------------------------------------------------------------#    
            # GAN Loss
            # ------------------------------------------------------------------------------------------------------------#  
            # train encoder / decoder      
            source_label = 0
            target_label = 1
            
            D_feat_c = modelD1(F.softmax(city_feat,dim=1)) 
            D_feat_p = modelD1(F.softmax(erp_feat,dim=1))
            
            loss_adv_c1 = bce_loss(D_feat_c, torch.FloatTensor(D_feat_c.data.size()).fill_(target_label).to(args.local_rank))
            loss_adv_p1 = bce_loss(D_feat_p, torch.FloatTensor(D_feat_p.data.size()).fill_(source_label).to(args.local_rank))
            
            D_feat_t_c = modelD2(F.softmax(pseudo_tangent_feat,dim=1)) 
            D_feat_t_p = modelD2(F.softmax(tangent_feat,dim=1))
            
            loss_adv_c2 = bce_loss(D_feat_t_c, torch.FloatTensor(D_feat_t_c.data.size()).fill_(target_label).to(args.local_rank))
            loss_adv_p2 = bce_loss(D_feat_t_p, torch.FloatTensor(D_feat_t_p.data.size()).fill_(source_label).to(args.local_rank))
            
            loss_d1 = loss_adv_c1 + loss_adv_p1
            loss_d2 = loss_adv_c2 + loss_adv_p2
            
            # train Discriminator   
            D_feat_c_ = modelD1(F.softmax(city_feat.detach(),dim=1)) 
            D_feat_p_ = modelD1(F.softmax(erp_feat.detach(),dim=1))
            
            loss_adv_c_ = bce_loss(D_feat_c_, torch.FloatTensor(D_feat_c_.data.size()).fill_(source_label).to(args.local_rank))
            loss_adv_p_ = bce_loss(D_feat_p_, torch.FloatTensor(D_feat_p_.data.size()).fill_(target_label).to(args.local_rank))
            
            D_feat_t_c_ = modelD1(F.softmax(pseudo_tangent_feat.detach(),dim=1)) 
            D_feat_t_p_ = modelD1(F.softmax(tangent_feat.detach(),dim=1))
            
            loss_adv_t_c_ = bce_loss(D_feat_t_c_, torch.FloatTensor(D_feat_t_c_.data.size()).fill_(source_label).to(args.local_rank))
            loss_adv_t_p_ = bce_loss(D_feat_t_p_, torch.FloatTensor(D_feat_t_p_.data.size()).fill_(target_label).to(args.local_rank))
            
            loss_d_1 = loss_adv_c_ + loss_adv_p_
            loss_d_2 = loss_adv_t_c_ + loss_adv_t_p_
            loss_d_ = loss_d_1 + loss_d_2
            # Supervised Loss
            # ------------------------------------------------------------------------------------------------------------#        
            loss_sup_1 = criterion_sup(city_pred,s_gt)
            writer.add_scalar('Model1 Sup Loss',loss_sup_1,it)
            loss_sup_2 = criterion_sup(pseudo_tangent_pred,pseudo_tangent_label)
            writer.add_scalar('Model2 Sup Loss',loss_sup_2,it)
            # Contrastive Loss
            # ------------------------------------------------------------------------------------------------------------#        
            loss = InfoNCE()
            loss_total_1, loss_total_2 = 0.0, 0.0
            upper_feat = batch_erp2tangent(erp_feat, 7) 
            upper_feat = upper_feat.to(args.local_rank).flatten(2)
            tangent_feat = tangent_feat.flatten(2)
            for i in range(49):
                loss_i = loss(F.log_softmax(upper_feat[:,:,i]), F.log_softmax(tangent_feat[:,:,i].detach()))
                loss_total_1 += loss_i
            loss_contrastive_1 = loss_total_1 / 49
            writer.add_scalar('Model1 Contrastive Loss',loss_contrastive_1,it)
            for i in range(49):
                loss_i = loss(F.log_softmax(tangent_feat[:,:,i]), F.log_softmax(upper_feat[:,:,i].detach()))
                loss_total_2 += loss_i
            loss_contrastive_2 = loss_total_2 / 49
            writer.add_scalar('Model2 Contrastive Loss',loss_contrastive_2,it)
            # Consistency Loss
            # ------------------------------------------------------------------------------------------------------------#        
            loss_con_1 = kl_loss(F.log_softmax(tangent_proj.permute(0,2,3,1)),F.log_softmax(tangent_pred.permute(0,2,3,1).detach()))
            writer.add_scalar('Model1 Con Loss',loss_con_1,it)
            loss_con_2 = kl_loss(F.log_softmax(tangent_pred.permute(0,2,3,1)),F.log_softmax(tangent_proj.permute(0,2,3,1).detach()))
            writer.add_scalar('Model2 Con Loss',loss_con_2,it)
            # Model Total Loss
            # ------------------------------------------------------------------------------------------------------------#        
            loss_1 = loss_sup_1 + get_current_consistency_weight(epoch, max_epoch) * args.alpha * loss_con_1 + loss_d1 + loss_contrastive_1
            loss_2 = loss_sup_2 + get_current_consistency_weight(epoch, max_epoch) * args.alpha * loss_con_2 + loss_d2 + loss_contrastive_2
            # Print Loss
            # ------------------------------------------------------------------------------------------------------------#        
            if it % pass_length == 0:
                if dist.get_rank() == 0:
                    print(f'it:{it};Model1 Total loss: {loss_1:.4f}')
                    print(f'it:{it};Model1 Sup loss: {loss_sup_1:.4f}')
                    print(f'it:{it};Model1 Consistency loss: {get_current_consistency_weight(epoch, max_epoch) * args.alpha * loss_con_2:.4f}')
                    print(f'it:{it};Model1 Contrastive loss: {loss_contrastive_1:.4f}')
                    
                    print(f'it:{it};Model2 Total loss: {loss_2:.4f}')
                    print(f'it:{it};Model2 Sup loss: {loss_sup_2:.4f}')
                    print(f'it:{it};Model2 Consistency loss: {get_current_consistency_weight(epoch, max_epoch) * args.alpha * loss_con_2:.4f}')
                    print(f'it:{it};Model2 Contrastive loss: {loss_contrastive_2:.4f}')
            # Model Optimization
            # ------------------------------------------------------------------------------------------------------------#        
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss_1.backward()
            loss_2.backward()
            optimizer1.step()
            optimizer2.step()
            # Discriminator Optimization
            # ------------------------------------------------------------------------------------------------------------#        
            optimizerD1.zero_grad()
            optimizerD2.zero_grad()
            loss_d_.backward()
            optimizerD1.step()
            optimizerD2.step()
            # Learning Rate 
            # ------------------------------------------------------------------------------------------------------------#        
            base_lr = args.lr
            if it <= 1500:
                lr_ = base_lr * (it / 1500)
                for param_group in optimizer1.param_groups:
                    param_group['lr'] = lr_ 
                for param_group in optimizer2.param_groups:
                    param_group['lr'] = lr_ 
            else:
                lr_ = adjust_learning_rate_poly(optimizer1,it - 1500,args.iterations,args.lr,1)
                lr_ = adjust_learning_rate_poly(optimizer2,it - 1500,args.iterations,args.lr,1)
            
            if it <= 1500:
                lr_d = d_lr * (it / 1500)
                for param_group in optimizerD1.param_groups:
                    param_group['lr'] = lr_d 
                for param_group in optimizerD2.param_groups:
                    param_group['lr'] = lr_d 
            else:
                lr_ = adjust_learning_rate_poly(optimizerD1,it - 1500,args.iterations,d_lr,1)
                lr_ = adjust_learning_rate_poly(optimizerD2,it - 1500,args.iterations,d_lr,1)
            # Validation
            # ------------------------------------------------------------------------------------------------------------#        
            if it % pass_length == 0 or it == 1:
                if it != 1:
                    epoch += 1
                    #==========================================Epochs_time==========================================#
                    time_elapsed = time.time() - since
                    print('Epoch cost {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
                miou_metrics_1 = mIOUMetrics(num_classes,255,args.local_rank)
                if dist.get_rank() == 0:
                    model1.eval()
                    since = time.time()
                    #==========================================model_eval==========================================#
                    with torch.no_grad():    
                        print(f'[Validation it: {it}] lr: {lr_}')
                       #==========================================model1_pass_eval==========================================#
                        val_mIOU_final = 0.0
                        total_val_mIOU = 0.0
                        for i, (image,label) in enumerate(pass_val_loader):
                                image, label = image.to(args.local_rank), label.to(args.local_rank)
                                pred, _ = model1(image)
                                miou_metrics_1.update(pred,label)
                                val_mIOU = miou_metrics_1.get_mIOU()
                                total_val_mIOU += val_mIOU
                        val_mIOU_final = total_val_mIOU/len(pass_val_loader)
                        miou_metrics_1.reset()
                        writer.add_scalar('model1 val mIOU',val_mIOU_final, epoch)
                        if val_mIOU_final > best_performance:
                            best_performance = val_mIOU_final
                            torch.save(model1.module.state_dict(),save_path+"/best.pth")
                        print('epoch:',epoch,'model1 val_mIOU:',val_mIOU_final, 'best:', best_performance)
                    #==========================================Evaluate_time==========================================#
                    time_elapsed = time.time() - since
                    print('Validate cost {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    print('file name: ', __file__)
    setup_seed(1234)
    main()
     
    
