import os
# import time
import numpy as np
import random
# from threading import Thread
# from scipy.io import loadmat
#from skvideo.io import vread
# import pdb
import torch
from torch.utils.data import Dataset
import pickle
import cv2
import sys
import av





class UCF101DataLoader(Dataset):
    'Prunes UCF101-24 data'
    def __init__(self, name, clip_shape, file_id, use_random_start_frame=False):
        self._dataset_dir = './data'

        if name == 'train':
            self.vid_files = self.get_det_annots_prepared(file_id)
            self.shuffle = True
            self.name = 'train'

        else:
            self.vid_files = self.get_det_annots_test_prepared(file_id)
            self.shuffle = False
            self.name = 'test'

        self._use_random_start_frame = use_random_start_frame
        self._height = clip_shape[0]
        self._width = clip_shape[1]
        self._size = len(self.vid_files)
        self.indexes = np.arange(self._size)
            

    def get_det_annots_prepared(self, file_id):
        import pickle
        
        training_annot_file = "./data_subset_pkl_files/"+ file_id
        
        with open(training_annot_file, 'rb') as tr_rid:
            training_annotations = pickle.load(tr_rid)
        print("Training samples from :", training_annot_file)
        
        return training_annotations
        
        
    def get_det_annots_test_prepared(self, file_id):
        import pickle    
        file_id = "test_annots.pkl"
        testing_anns  ="./data_subset_pkl_files/" + file_id
        with open(testing_anns, 'rb') as ts_rid:
            testing_annotations = pickle.load(ts_rid)
            
        return testing_annotations    
    
    


    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.vid_files)

    def __getitem__(self, index):
        
        depth = 8
        video_rgb = np.zeros((depth, self._height, self._width, 3))     # 8, 224,224, 3
        label_cls = np.zeros((depth, self._height, self._width, 1))     # FG/BG or actor (person only) for this dataset
        
        v_name, anns= self.vid_files[index]

        clip, bbox_clip, label, annot_frames, labeled_vid ,x_y_min= self.load_video(v_name, anns)
        # print("x_y_min:{}".format(x_y_min))
        sys.stdout.flush() 
        if clip is None:
            video_rgb = np.transpose(video_rgb, [3, 0, 1, 2])  #moving channels to first position
            video_rgb = torch.from_numpy(video_rgb)            
            label_cls = np.transpose(label_cls, [3, 0, 1, 2])
            label_cls = torch.from_numpy(label_cls)
            labeled_vid = 0
            sample = {'data':video_rgb,'loc_msk':label_cls,'action':torch.Tensor([0]), 'aug_data': video_rgb, 'label_vid':labeled_vid}
            return sample

        #vlen = clip.shape[0]
        vlen, clip_h, clip_w, _ = clip.shape
        # clip_h = int(270)
        clip_w = int(480)
        vskip = 2
        
        if len(annot_frames) == 1:
            selected_annot_frame = annot_frames[0]
        else:
            if len(annot_frames) <= 0:
                print('annot index error for', v_name, ', ', len(annot_frames), ', ', annot_frames)
                video_rgb = np.transpose(video_rgb, [3, 0, 1, 2])  #moving channels to first position
                video_rgb = torch.from_numpy(video_rgb)            
                label_cls = np.transpose(label_cls, [3, 0, 1, 2])
                label_cls = torch.from_numpy(label_cls)
                labeled_vid = 0
                sample = {'data':video_rgb,'loc_msk':label_cls,'action':torch.Tensor([0]), 'aug_data': video_rgb, 'label_vid':labeled_vid}

                return sample
            annot_idx = np.random.randint(0,len(annot_frames))
            selected_annot_frame = annot_frames[annot_idx]
            # print(f'selected Frame: {selected_annot_frame}')
        start_frame = selected_annot_frame - int((depth * vskip)/2)

        if start_frame < 0:
            vskip = 1
            start_frame = selected_annot_frame - int((depth * vskip)/2)
            if start_frame < 0:
                start_frame = 0
                vskip = 1
        if selected_annot_frame >= vlen:
            video_rgb = np.transpose(video_rgb, [3, 0, 1, 2])  #moving channels to first position
            video_rgb = torch.from_numpy(video_rgb)            
            label_cls = np.transpose(label_cls, [3, 0, 1, 2])
            label_cls = torch.from_numpy(label_cls)
            labeled_vid = 0
            sample = {'data':video_rgb,'loc_msk':label_cls,'action':torch.Tensor([0]), 'aug_data': video_rgb, 'label_vid': labeled_vid}
            return sample
        if start_frame + (depth * vskip) >= vlen:
            start_frame = vlen - (depth * vskip)
        
        # frame index to chose - 0, 2, 4, ..., 16
        span = (np.arange(depth)*vskip)

        # frame_ids
        span += start_frame
        video = clip[span]
        bbox_clip = bbox_clip[span]

        # closest_fidx = np.argmin(np.abs(span-selected_annot_frame))
        
        if self.name == 'train':
            # take random crops for training
            if clip_w-x_y_min[0]+x_y_min[2]>x_y_min[0]:
                start_pos_w =random.randint(0,x_y_min[0]) #self._width)
            else:
                start_pos_w =random.randint(x_y_min[0]+x_y_min[2],clip_w)-224 #self._width)
            start_pos_h =random.randint(0,x_y_min[1]) #self._height)         
        else:
            # center crop for validation
            start_pos_h = random.randint(0,x_y_min[1])
            start_pos_w = int(x_y_min[0]+(x_y_min[2]) / float(2)-112)
        
        for j in range(video.shape[0]):
            img = video[j]
            img = np.dstack((img[:,:,2],img[:,:,1],img[:,:,0]))
            img = cv2.resize(img,(480,270))
            img = img[start_pos_h:start_pos_h+224, start_pos_w:start_pos_w+224,:]
            img = cv2.resize(img, (self._height,self._width), interpolation=cv2.INTER_LINEAR)
            img = img / 255.
            video_rgb[j] = img
            
            bbox_img = bbox_clip[j]
            bbox_img = bbox_img[start_pos_h:start_pos_h+224, start_pos_w:start_pos_w+224,:]
            bbox_img = cv2.resize(bbox_img, (self._height,self._width), interpolation=cv2.INTER_LINEAR)
            label_cls[j, bbox_img > 0, 0] = 1.
                       
        
        horizontal_flipped_video = video_rgb[:, :, ::-1, :]
        # horizontal_flipped_label_cls = label_cls[:,:,::-1,:]

        video_rgb = np.transpose(video_rgb, [3, 0, 1, 2])  #moving channels to first position
        video_rgb = torch.from_numpy(video_rgb)
        
        label_cls = np.transpose(label_cls, [3, 0, 1, 2])
        label_cls = torch.from_numpy(label_cls)

        horizontal_flipped_video = np.transpose(horizontal_flipped_video, [3, 0, 1, 2])
        horizontal_flipped_video = torch.from_numpy(horizontal_flipped_video.copy())
        
        action_tensor = torch.Tensor([label])

        sample = {'data':video_rgb,'loc_msk':label_cls,'action':action_tensor, "aug_data":horizontal_flipped_video, "label_vid": labeled_vid}

        return sample


    def load_video(self, video_name, annotations):
        video_dir = os.path.join(self._dataset_dir, 'fudan/%s' % video_name).replace('\\','/')
        container = None
        try:
            container = av.open(video_dir)
        except Exception as e:
            print(
                "Failed to load video from {} with error{}".format(video_dir,e)
            )
        video = None
        if container is not None:
            videos = [img.to_rgb().to_ndarray() for img in container.decode(video = 0)]
            video = np.stack(videos,axis=0)
        # try:
        #     video = vread(str(video_dir)) # Reads in the video into shape (F, H, W, 3)
        #     # print(video.shape)
        # except:
        #     print('Error:', str(video_dir))
        #     return None, None, None, None, None

        # creates the bounding box annotation at each frame
        n_frames =len(video)
        h = int(270)
        w = int(480)
        bbox = np.zeros((n_frames, h, w, 1), dtype=np.uint8)
        label = -1
        labeled_vid = -1
        #multi frame mode
        # annot_idx = 0
        if len(annotations) > 1:
            annot_idx = np.random.randint(0,len(annotations))
        multi_frame_annot = []      # annotations[annot_idx][4]
        bbox_annot = np.zeros((n_frames, h, w, 1), dtype=np.uint8)
        x_y_min =[]
        for ann in annotations:     
            multi_frame_annot.extend(ann[4])
            start_frame, end_frame, label, labeled_vid = ann[0], ann[1], ann[2], ann[5]
    
            collect_annots = []
            for f in range(start_frame, min(n_frames, end_frame+1)):
                try:
                    x, y, w, h = ann[3][f-start_frame]
                    x = int(x/float(2))
                    y = int(y/float(2))
                    w = int(w/float(2))
                    h = int(h/float(2))
                    bbox[f, y:y+h, x:x+w, :] = 1
                    if f in ann[4]:
                        collect_annots.append([x,y,w,h])
                except:
                    print('ERROR LOADING ANNOTATIONS')
                    print(start_frame, end_frame)
                    print(video_dir)
                    exit()
            # Expect to have collect_annots with same length as annots for this set 
            # [ c, c, c, c, c, c ....]
            select_annots = ann[4]
            select_annots.sort()
            if len(collect_annots) == 0:
                continue
                
            # x_min, y_min, width, height
            [x, y, w, h] = collect_annots[0]
            x_y_min.extend(collect_annots[0])
            
        multi_frame_annot = list(set(multi_frame_annot))
        if self.name == 'train':
            return video, bbox, label, multi_frame_annot, labeled_vid,x_y_min
        else:
            return video, bbox, label, multi_frame_annot, labeled_vid,x_y_min



if __name__ =="__main__":
    # 'data':video_rgb,'loc_msk':label_cls,'action':action_tensor, "aug_data":horizontal_flipped_video, "label_vid": labeled_vid
    data = UCF101DataLoader('train', [224, 224], file_id='train_annots_label.pkl', use_random_start_frame=False)
    print(data[0]['action'])
