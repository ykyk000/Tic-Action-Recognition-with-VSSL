from cProfile import label
import os
import random
import cv2
import pickle
import numpy as np
import argparse
# data_path1 = './data_subset_pkl_files/'+'train_annots_unlabel.pkl'
# data_path2 = './data_subset_pkl_files/'+'test_annots.pkl'
# # # # for da in sorted(os.listdir(data_path)):
# # # #     da_path = os.path.join(data_path,da).replace('\\','/')
# with open(data_path1, 'rb') as tr_rid:
#         train1 = pickle.load(tr_rid)
# with open(data_path2, 'rb') as tr_rid:
#         train2 = pickle.load(tr_rid)
# print(len(train1))
# print(len(train2))
# print(len(train2)+len(train1))

def make_sup_unsup(video_mes,rate):
    lenth = len(video_mes)
    sup = int (rate*lenth)
    sup_number = random.sample(range(0,lenth),int(sup))
    sup_video = [video_mes[ord] for ord in sup_number]
    unsup_video  = [li for i ,li in enumerate(video_mes) if i not in sup_number]
    new_sup = make_video(sup_video,True)
    new_unsup = make_video(unsup_video,False)
    return new_sup,new_unsup
  

def make_video(video_data,sup_unsup =True):
    lists = []
    for video in video_data:
        video_adr = video[0].split('/')
        video_adr = os.path.join(video_adr[2],video_adr[3],video_adr[-1]).replace('\\','/')
        annotions = list(video[-1])
        annotions[3] =np.array(annotions[3])
        if sup_unsup:
            annotions.append(int(1))
        else:
            annotions.append(int(0))
            
        new_video = (video_adr,[tuple(annotions)])
        lists.append(new_video)
    return lists


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='make data train or test')
    parser.add_argument('--mode', type=str, default='train', help='train or test',choices=['train','test'])
    parser.add_argument('--rate', type=float, default=0.2, help='percentage of supervised train sets',choices=[0.05,0.1,0.15,0.2])
    parser.add_argument('--ori_train_path', type=str, default='./project/train', help='Path of Ori. train datasets')
    parser.add_argument('--ori_test_path', type=str, default='./project/test', help='Path of Ori. test datasets')
    parser.add_argument('--out_train_path', type=str, default='./data_subset_pkl_files', help='Path of output train datasets')
    parser.add_argument('--out_test_path', type=str, default='./data_subset_pkl_files', help='Path of output test datasets')
    args = parser.parse_args()
    if args.mode == 'train':
        sup_list = []
        unsup_list = []
        for da in sorted(os.listdir(args.ori_train_path)):
            da_path = os.path.join(args.ori_train_path,da).replace('\\','/')
            with open(da_path, 'rb') as tr_rid:
                video_mes = pickle.load(tr_rid)
            sup ,unsup = make_sup_unsup(video_mes,args.rate)
            sup_list.extend(sup)
            unsup_list.extend(unsup)
        save_label_path = args.out_train_path+'/'+str(args.rate)+'_train_annots_label.pkl'
        save_unlabel_path = args.out_train_path+'/'+str(1-args.rate)+'_train_annots_unlabel.pkl'
        if not os.path.exists(save_label_path):
            with open(save_label_path,'wb') as f:
                pickle.dump(sup_list,f) 
        if not os.path.exists(save_unlabel_path):
            with open(save_unlabel_path,'wb') as f:
                pickle.dump(unsup_list,f) 
        print('make train datasets successfully !!!!!')
    elif args.mode == 'test':
        data_test = []
        for da in sorted(os.listdir(args.ori_test_path)):
            da_path = os.path.join(args.ori_test_path,da).replace('\\','/')
            with open(da_path, 'rb') as tr_rid:
                video_mes = pickle.load(tr_rid)
            test_da = make_video(video_mes,False)    
            data_test.extend(test_da)
        save_path = args.out_test_path+'/'+'test_annots.pkl'
        if not os.path.exists(save_path):
            with open(save_path,'wb') as f:
                pickle.dump(data_test,f) 
        print('make test datasets successfully !!!!!')
    else:
        print('error !!!!')