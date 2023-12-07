import sys
import os
import torch
import time
import random
import argparse
# import datetime
import numpy as np
import pandas as pd
# debugging purposes
#import imageio

import warnings
warnings.filterwarnings("ignore")

import torch.nn as nn
from torch import optim
# from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from tqdm import tqdm
#from tensorboardX import SummaryWriter

from datasets.ucf_dataloader import UCF101DataLoader
from models.capsules_ucf101 import CapsNet

from utils.losses import SpreadLoss, DiceLoss, weighted_mse_loss
from utils.metrics import get_accuracy, IOU2
from utils.helpers import measure_pixelwise_var_v2, measure_pixelwise_gradient
from utils import ramp_ups

def val_model_interface(minibatch):
    data = minibatch['data'].type(torch.cuda.FloatTensor)
    action = minibatch['action'].cuda()
    segmentation = minibatch['loc_msk']
    empty_vector = torch.zeros(action.shape[0]).cuda()

    output, predicted_action, _ = model(data, action, empty_vector, 0, 0)
    
    class_loss, abs_class_loss = criterion_cls(predicted_action, action)
    loss1 = criterion_seg_1(output, segmentation.float().cuda())
    loss2 = criterion_seg_2(output, segmentation.float().cuda())
    
    loc_loss = loss1 + loss2
    total_loss =  loc_loss + class_loss
    return (output, predicted_action, segmentation, action, total_loss, loc_loss, class_loss)
    

def train_model_interface(args, label_minibatch, unlabel_minibatch, epoch, wt_ramp):
    # read data
    label_data = label_minibatch['data'].type(torch.cuda.FloatTensor)
    fl_label_data = label_minibatch['aug_data'].type(torch.cuda.FloatTensor)

    unlabel_data = unlabel_minibatch['data'].type(torch.cuda.FloatTensor)
    fl_unlabel_data = unlabel_minibatch['aug_data'].type(torch.cuda.FloatTensor)

    label_action = label_minibatch['action'].cuda()
    unlabel_action = unlabel_minibatch['action'].cuda()

    label_segmentation = label_minibatch['loc_msk']
    unlabel_segmentation = unlabel_minibatch['loc_msk']

    # concat data and shuffle
    concat_data = torch.cat([label_data, unlabel_data], dim=0)
    concat_fl_data = torch.cat([fl_label_data, fl_unlabel_data], dim=0)
    concat_action = torch.cat([label_action, unlabel_action], dim=0)
    concat_seg = torch.cat([label_segmentation, unlabel_segmentation], dim=0)

    sup_vid_labels = label_minibatch['label_vid']
    unsup_vid_labels = unlabel_minibatch['label_vid']
    concat_labels = torch.cat([sup_vid_labels, unsup_vid_labels], dim=0).cuda()
    random_indices = torch.randperm(len(concat_labels))

    concat_data = concat_data[random_indices, :, :, :, :]

    concat_fl_data = concat_fl_data[random_indices, :, :, :,:]

    concat_action = concat_action[random_indices]

    concat_labels = concat_labels[random_indices]

    concat_seg = concat_seg[random_indices, :, :, :, :]

    labeled_vid_index = torch.where(concat_labels==1)[0]    

    # passing inputs to models
    output, predicted_action, feat = model(concat_data, concat_action, concat_labels, epoch, args.thresh_epoch)
    flip_op, _, aug_feat = model(concat_fl_data, concat_action, concat_labels, epoch, args.thresh_epoch)

    # SEG LOSS SUPERVISED
    labeled_op = output[labeled_vid_index]
    labeled_seg_data = concat_seg[labeled_vid_index]
    loc_loss_1 = criterion_seg_1(labeled_op, labeled_seg_data.float().cuda())
    loc_loss_2 = criterion_seg_2(labeled_op, labeled_seg_data.float().cuda())
    
    # Classification loss SUPERVISED
    labeled_cls = concat_action[labeled_vid_index]
    labeled_pred_action = predicted_action[labeled_vid_index]
    class_loss, abs_class_loss = criterion_cls(labeled_pred_action, labeled_cls)
    loss_const_cls = js_div(feat,aug_feat)

    flipped_pred_seg_map = torch.flip(flip_op, [4])
    equal_wt = torch.ones_like(output, dtype=torch.double)
    equal_wt = equal_wt.type(torch.cuda.FloatTensor)

    loss_wt_simple_l2 = weighted_mse_loss(flipped_pred_seg_map, output, equal_wt)

    if args.bv:

        batch_variance_clck = measure_pixelwise_var_v2(output, torch.flip(flipped_pred_seg_map, [2]), frames_cnt=args.n_frames, use_sig_output=args.predict_maps)
        batch_variance_anticlck = measure_pixelwise_var_v2(torch.flip(output, [2]), flipped_pred_seg_map, frames_cnt=args.n_frames, use_sig_output=args.predict_maps)
        
        batch_variance_clck = batch_variance_clck.type(torch.cuda.FloatTensor)
        batch_variance_anticlck = batch_variance_anticlck.type(torch.cuda.FloatTensor)

        loss_wt_var_1 = weighted_mse_loss(flipped_pred_seg_map, output, batch_variance_clck)
        loss_wt_var_2 = weighted_mse_loss(flipped_pred_seg_map, output, torch.flip(batch_variance_anticlck, [2]))
        
        
        total_seg_cons_loss_1 = (wt_ramp * (loss_wt_var_1 + loss_wt_var_2)) + ((1 - wt_ramp) * loss_wt_simple_l2)


    if args.gv:
        batch_grad = measure_pixelwise_gradient(output, conf_thresh_lower=args.lower_thresh, conf_thresh_upper=args.upper_thresh)
        batch_grad = batch_grad.type(torch.cuda.FloatTensor)
        # ----------自己的方法------------------
        batch_grad_aug = measure_pixelwise_gradient(flipped_pred_seg_map, conf_thresh_lower=args.lower_thresh, conf_thresh_upper=args.upper_thresh)
        batch_grad_aug= batch_grad.type(torch.cuda.FloatTensor)
        loss_wt_grad = consistency_criterion(batch_grad,batch_grad_aug)
    
    if args.bv and args.gv:
        total_seg_cons_loss = args.bv_wt * total_seg_cons_loss_1 + args.gv_wt * loss_wt_grad
    elif args.gv:
        total_seg_cons_loss = loss_wt_grad
    elif args.bv:
        total_seg_cons_loss = total_seg_cons_loss_1

    total_cons_loss = args.wt_cons_vg*total_seg_cons_loss+args.wt_cons_cls*loss_const_cls
    
    loc_loss = loc_loss_1 + loc_loss_2
    # label_loss = loc_loss +class_loss
    total_loss = args.wt_loc * loc_loss + args.wt_cls * class_loss + args.wt_cons * total_cons_loss

    return (output, predicted_action, concat_seg, concat_action, total_loss, loc_loss, class_loss,loss_const_cls,total_seg_cons_loss,total_cons_loss)




def train(args, model, labeled_train_loader, unlabeled_train_loader, optimizer, epoch, ramp_wt):
    model.train(mode=True)
    model.training = True
    total_loss = []
    accuracy_1 = []
    accuracy_5 = []
    loc_loss = []
    class_loss = []
    total_seg_cons_loss=[]
    loss_const_cls = []
    class_consistency_loss = []
    label_list = []
    pred_list = []
    ROC_list = []
    steps = len(unlabeled_train_loader)

    start_time = time.time()

    labeled_iterloader = iter(labeled_train_loader)
    
    for batch_id, unlabel_minibatch  in enumerate(unlabeled_train_loader):
    
        optimizer.zero_grad()

        try:
            label_minibatch = next(labeled_iterloader)

        except StopIteration:
            labeled_iterloader = iter(labeled_train_loader)
            label_minibatch = next(labeled_iterloader)

        output, predicted_action, segmentation, action, loss, s_loss, c_loss,loss_cls,seg_cons_loss, cc_loss =\
         train_model_interface(args, label_minibatch, unlabel_minibatch, epoch, ramp_wt(epoch))

        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())
        loc_loss.append(s_loss.item())
        class_loss.append(c_loss.item())
        loss_const_cls.append(loss_cls.item())
        total_seg_cons_loss.append(seg_cons_loss.item())
        class_consistency_loss.append(cc_loss.item())
        top_1,top_5= get_accuracy(predicted_action, action)
        accuracy_1.append(top_1)
        accuracy_5.append(top_5)
        pred_list.extend(attend_li(predicted_action,action)[0])
        label_list.extend(attend_li(predicted_action,action)[1])
        ROC_list.append([np.array(action.squeeze().cpu(),dtype=np.int8),predicted_action.squeeze().detach().cpu().numpy()])
        if (batch_id + 1) % args.pf == 0:
            r_total = np.array(total_loss).mean()
            r_seg = np.array(loc_loss).mean()
            r_class = np.array(class_loss).mean()
            r_const = np.array(class_consistency_loss).mean()
            r_acc = np.array(accuracy_1).mean()
            print(f'[TRAIN] epoch-{epoch:0{len(str(args.epochs))}}/{args.epochs}, batch-{batch_id+1:0{len(str(steps))}}/{steps},' \
                  f'loss-{r_total:.3f}, acc-{r_acc:.3f}' \
                  f'\t [LOSS ] cls-{r_class:.3f}, seg-{r_seg:.3f}, const-{r_const:.3f}')

            sys.stdout.flush()

    end_time = time.time()
    train_epoch_time = end_time - start_time
    print("Training time: ", train_epoch_time)
    r_class = np.array(class_loss).mean()
    r_const = np.array(class_consistency_loss).mean()
    r_acc_1 = np.array(accuracy_1).mean()
    r_acc_5 = np.array(accuracy_5).mean()
    r_const_cls = np.array(loss_const_cls).mean() 
    r_const_seg = np.array(total_seg_cons_loss).mean()
    train_total_loss = np.array(total_loss).mean()
    precision,recall,f1_score = generate_confu_marix(pred_list,label_list,save_pth ='./results',epoch = epoch)
    import pickle 
    with open('./results/ROC.pkl','ab') as f:
        pickle.dump([epoch,ROC_list],f)
    print('Train Epoch:{}/{}  Total_loss:{}  class_loss:{}  class_consistency_loss:{}  acc@1:{}% acc@5:{} %'.format(epoch,args.epochs\
        ,train_total_loss,r_class,r_const,r_acc_1*100,r_acc_5*100))
    sys.stdout.flush() 
    return train_total_loss,r_const_cls,r_const_seg,r_class,r_const,r_acc_1,r_acc_5,precision,recall,f1_score


def validate(model, val_data_loader, epoch):
    steps = len(val_data_loader)
    model.eval()
    model.training = False
    total_loss = []
    accuracy_1 = []
    accuracy_5 =[]
    loc_loss = []
    class_loss = []
    total_IOU = 0
    validiou = 0
    print('validating...')
    start_time = time.time()
    
    with torch.no_grad():
        
        for batch_id, minibatch in enumerate(val_data_loader):
            
            output, predicted_action, segmentation, action, loss, s_loss, c_loss = val_model_interface(minibatch)
            total_loss.append(loss.item())
            loc_loss.append(s_loss.item())
            class_loss.append(c_loss.item())
            top_1,top_5= get_accuracy(predicted_action, action)
            accuracy_1.append(top_1)
            accuracy_5.append(top_5)

            maskout = output.cpu()
            maskout_np = maskout.data.numpy()
            # utils.show(maskout_np[0])

            # use threshold to make mask binary
            maskout_np[maskout_np > 0] = 1
            maskout_np[maskout_np < 1] = 0
            # utils.show(maskout_np[0])

            truth_np = segmentation.cpu().data.numpy()
            for a in range(minibatch['data'].shape[0]):
                iou = IOU2(truth_np[a], maskout_np[a])
                if iou == iou:
                    total_IOU += iou
                    validiou += 1
                else:
                    print('bad IOU')
    
    val_epoch_time = time.time() - start_time
    print("Validation time: ", val_epoch_time)
    
    r_total = np.array(total_loss).mean()
    r_seg = np.array(loc_loss).mean()
    r_class = np.array(class_loss).mean()
    r_acc_1 = np.array(accuracy_1).mean()
    r_acc_5 = np.array(accuracy_5).mean()
    average_IOU = total_IOU / validiou
    print(f'[VAL] epoch-{epoch}, loss-{r_total:.3f}, acc_@1-{r_acc_1:.3f} ,acc_@5-{r_acc_5:.3f} [IOU ] {average_IOU:.3f}')
    sys.stdout.flush()
    return r_total,r_acc_1,r_acc_5



def js_div(p_out,q_out,get_softmax = True):
    KLDivloss = torch.nn.KLDivLoss(reduction='batchmean')
    if get_softmax:
        p_out = torch.nn.Softmax(dim = -1)(p_out)
        q_out = torch.nn.Softmax(dim =-1)(q_out)
    log_mean_output = ((p_out + q_out )/2).log()
    return (KLDivloss(log_mean_output, p_out) + KLDivloss(log_mean_output, q_out))/2
def parse_args():
    parser = argparse.ArgumentParser(description='loc var const')
    parser.add_argument('--bs', type=int, default=8, help='mini-batch size')
    parser.add_argument('--epochs', type=int, default=20, help='number of total epochs to run')
    parser.add_argument('--model_name', type=str, default='i3d', help='model name')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--pf', type=int, default=50, help='print frequency every batch')
    parser.add_argument('--pretrained', type=str, default="i3d", help='loading pretrained model')
    parser.add_argument('--loc_loss', type=str, default='dice', help='dice or iou loss')
    parser.add_argument('--exp_id', type=str, default='debug', help='experiment name')

    parser.add_argument('--pkl_file_label', type=str, default='train_annots_label.pkl', help='label subset')
    parser.add_argument('--pkl_file_unlabel', type=str, default='train_annots_unlabel.pkl', help='unlabele subset')

    parser.add_argument('--const_loss', type=str, default='l2', help='consistency loss type')
    parser.add_argument('--wt_loc', type=float, default=1, help='segmentation loss weight')
    parser.add_argument('--wt_cls', type=float, default=1, help='Classification loss weight')
    parser.add_argument('--wt_cons', type=float, default=0.3, help='class consistency loss weight')
    parser.add_argument('--wt_cons_cls', type=float, default=0.3, help='const cls loss weight')
    parser.add_argument('--wt_cons_vg', type=float, default=0.7, help='const var and gridient loss weight')
    parser.add_argument('--seed', type=int, default=47, help='seed for initializing training.')

    parser.add_argument('--thresh_epoch', type=int, default=11, help='thresh epoch to introduce pseudo labels')
    parser.add_argument('--workers', type=int, default=8, help='num workers')

    parser.add_argument('--n_frames', type=int, default=3, help='batch variance frames number.')
    parser.add_argument('--bv',type=bool,default=True, help='use batch variance')
    parser.add_argument('--predict_maps', type=bool,default=True, help='use sigmoid outputs')    
    parser.add_argument('--bv_wt', type=float, default=0.5, help='batch variance weight')
    parser.add_argument('--cyclic', type=bool,default=True, help='use batch variance')

    parser.add_argument('--gv',type=bool,default=True, help='use grad variance')
    parser.add_argument('--lower_thresh', type=float, default=None, help='lower conf thresh')
    parser.add_argument('--upper_thresh', type=float, default=None, help='upper conf thresh')
    parser.add_argument('--gv_wt', type=float, default=0.5, help='grad variance weight')

    args = parser.parse_args()
    return args
def generate_confu_marix(predict,target,average = 'macro',save_pth = None,epoch = None):
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import precision_score,f1_score,recall_score
    import pickle 
    target = np.array(target)
    predict = np.array(predict)

    order_num = target.argsort()

    label_up = target[order_num]  
    pred_up = predict[order_num]

    cm = confusion_matrix(label_up, pred_up)

    precision = precision_score(label_up, pred_up,average=average)

    recall = recall_score(label_up, pred_up,average=average)

    f1 = f1_score(label_up, pred_up,average=average)
    if save_pth:
        save_pth = save_pth+'/train.pkl'
        with open(save_pth,'ab') as f:
            if epoch:
                pickle.dump([epoch,label_up,pred_up,cm],f)
            else:
                pickle.dump([label_up,pred_up,cm],f)
    return (precision,recall,f1)
def parse_mxpkl(path):
    import pickle 
    path = path+'/train.pkl'
    all = []
    with open(path,'rb') as f: 
        while True:
            try:
                aa=pickle.load(f)
                all.append(aa)
            except EOFError:
                break
    return all
def attend_li(predict,target):
    import numpy as np

    pred = predict.argsort(dim=-1, descending=True)[:,:1].squeeze().cpu().numpy()

    label =  np.array(target.squeeze().cpu(),dtype=np.int8)
    return pred,label
if __name__ == '__main__':
    torch.cuda.set_device(1)
    args = parse_args()
    print(vars(args))
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device:{device}")
    # USE_CUDA = True if torch.cuda.is_available() else False
    # if torch.cuda.is_available() and not USE_CUDA:
    #     print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
    # TRAIN PARAMS
    TRAIN_BATCH_SIZE = args.bs
    VAL_BATCH_SIZE = 1
    N_EPOCHS = args.epochs
    LR = args.lr
    loc_loss_criteria = args.loc_loss


    labeled_trainset = UCF101DataLoader('train', [224, 224], file_id=args.pkl_file_label, use_random_start_frame=False)
    unlabeled_trainset = UCF101DataLoader('train', [224, 224], file_id=args.pkl_file_unlabel, use_random_start_frame=False)
    validationset = UCF101DataLoader('validation',[224, 224], file_id="test_annots.pkl", use_random_start_frame=False)
    
    print(len(labeled_trainset), len(unlabeled_trainset), len(validationset))


    # label train dataloader
    labeled_train_data_loader = DataLoader(
        dataset=labeled_trainset,
        batch_size=TRAIN_BATCH_SIZE//2,
        shuffle=True
    )

    # unlabel train dataloader
    unlabeled_train_data_loader = DataLoader(
        dataset=unlabeled_trainset,
        batch_size=(TRAIN_BATCH_SIZE)//2,
        shuffle=True
    )

    # validation dataloader
    val_data_loader = DataLoader(
        dataset=validationset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False
    )

    print(len(labeled_train_data_loader), len(unlabeled_train_data_loader), len(val_data_loader))
    
    # Load pretrained weights
    model = CapsNet()
    
    #if USE_CUDA:
    model = model.to(device)
    
    # define losses
    global criterion_cls
    global criterion_seg_1
    global criterion_seg_2
    global consistency_criterion
    criterion_cls = SpreadLoss(num_class=13, m_min=0.2, m_max=0.9)

    criterion_seg_1 = nn.BCEWithLogitsLoss(size_average=True)   # size_average will be deprecated use reduction=mean

    if loc_loss_criteria == 'dice':
        criterion_seg_2 = DiceLoss()

    # elif loc_loss_criteria == 'iou':
    #     criterion_seg_2 = IoULoss()

    else:
        print("wrong parameter recheck. Exiting the code !!!!")
        exit()
        
    if args.const_loss == 'jsd':
        consistency_criterion = torch.nn.KLDivLoss(size_average=False, reduce=False).cuda()

    elif args.const_loss == 'l2':
        consistency_criterion = nn.MSELoss()

    elif args.const_loss == 'l1':
        consistency_criterion = nn.L1Loss()

    else:
        print("no consistency criterion found. Exiting the code!!!")
        exit()


    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0, eps=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-7, patience=5, factor=0.1, verbose=True)

    ramp_wt = ramp_ups.exp_rampup(N_EPOCHS)

    exp_id = args.exp_id
    save_path = os.path.join('train_log_wts', exp_id)
    # model_save_dir = os.path.join(save_path,time.strftime('%m-%d-%H-%M'))
    # writer = SummaryWriter(model_save_dir)
    # if not os.path.exists(model_save_dir):
    #     os.makedirs(model_save_dir)

    model_save_dir = './results'
    # writer = SummaryWriter(model_save_dir)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    results = {'train_loss': [], 'const_cls': [],'const_v_g': [],'class_loss': [], 'loc_loss': [],'accuracy@1': [],'accuracy@5': [],\
        'precision':[],'recall':[],'f1_score':[],'val_loss': [],'val_acc@1': [],'val_acc@5': []}
    prev_best_val_loss = 10000
    prev_best_train_loss = 10000
    prev_best_val_loss_model_path = None
    prev_best_train_loss_model_path = None

    for e in tqdm(range(1, N_EPOCHS + 1)):
        
        train_loss,const_cls,const_vg, class_loss,class_consistency_loss,\
           accuracy_1,accuracy_5,precision,recall,f1_score= train(args, model, labeled_train_data_loader, unlabeled_train_data_loader, optimizer, e, ramp_wt)

        val_loss,val_acc_1,val_acc_5 = validate(model, val_data_loader, e)
        results['train_loss'].append(train_loss)
        results['const_cls'].append(const_cls)
        results['const_v_g'].append(const_vg)
        results['class_loss'].append(class_loss)
        results['loc_loss'].append(class_consistency_loss)
        results['accuracy@1'].append(accuracy_1)
        results['accuracy@5'].append(accuracy_5)
        results['precision'].append(precision)
        results['recall'].append(recall)
        results['f1_score'].append(f1_score)
        results['val_loss'].append(val_loss)
        results['val_acc@1'].append(val_acc_1)
        results['val_acc@5'].append(val_acc_5)
        data_frame = pd.DataFrame(data=results, index=range(1, e + 1))
        data_frame.to_csv('{}/{}_statistics.csv'.format(model_save_dir,'train'), index_label='epoch')
        if val_loss < prev_best_val_loss:
             print("Yay!!! Got the val loss down...")
             val_model_path = os.path.join(model_save_dir, f'best_model_val_loss_{e}.pth').replace('\\','/')
             torch.save(model.state_dict(), val_model_path)
             prev_best_val_loss = val_loss
             if prev_best_val_loss_model_path and e<20:
                 os.remove(prev_best_val_loss_model_path)
             prev_best_val_loss_model_path = val_model_path

        if train_loss < prev_best_train_loss:
             print("Yay!!! Got the train loss down...")
             train_model_path = os.path.join(model_save_dir, f'best_model_train_loss_{e}.pth')
             torch.save(model.state_dict(), train_model_path)
             prev_best_train_loss = train_loss
             if prev_best_train_loss_model_path and e<20:
                 os.remove(prev_best_train_loss_model_path)
             prev_best_train_loss_model_path = train_model_path
        scheduler.step(train_loss)