import os
import glob
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.nn import functional as F

from .datasets import register_dataset

import pickle

@register_dataset("mtl_aqa")
class MTL_AQA(Dataset):
    def __init__(self,
                is_training,
                split,
                data_root,
                video_dir,
                label_file,
                train_datafile,
                test_datafile,
                stride,
                window_size,
                frames_per_clip,
                max_seq_len,
                input_frame_size = None,
                crop_frame_size = None,
                with_dd = True,
                three_judge_score_scaling = False,  
                use_feats = None,
                feat_dir = None,
                ):   
        
        self.subset = split
        self.is_training = is_training

        self.use_feats = use_feats

        self.crop_frame_size = crop_frame_size

        self.with_dd = with_dd
        self.three_judge_score_scaling = three_judge_score_scaling

        

        if self.subset == 'test':
            self.transforms = transforms.Compose(
                    [
                    transforms.Resize(input_frame_size),
                    transforms.CenterCrop(crop_frame_size),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                    ]
                )
            

        elif self.subset == 'train':
            self.transforms = transforms.Compose(
                        [
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.Resize(input_frame_size),
                        transforms.RandomCrop(crop_frame_size),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                        ]
                    )

        else:
            raise ValueError("subset should be train or test")
        
        print(f"subset: {self.subset}, is_training: {self.is_training}")

        self.pil_2_tensor = transforms.ToTensor()
        self.stride = stride
        self.window_size = window_size
        self.frames_per_clip = frames_per_clip
        self.max_seq_len = max_seq_len

        # file paths
        self.data_root = data_root
        self.video_dir = os.path.join(self.data_root, video_dir)

        if self.use_feats:
            self.feat_dir = os.path.join(self.data_root, feat_dir)

        train_datafile_path = os.path.join(self.data_root, "info", train_datafile)
        test_datafile_path = os.path.join(self.data_root,"info", test_datafile)

        self.data_anno = self.read_pickle(os.path.join(self.data_root, "info", label_file))

        if self.subset == 'test':
            self.datalist = self._load_annotation(test_datafile_path)
        elif self.subset == 'train':
            self.datalist = self._load_annotation(train_datafile_path)
        else:
            raise ValueError("subset should be train or test")


    def _load_annotation(self, pkl_file):

        data_list = self.read_pickle(pkl_file)
        processed_data_list = []

        for video_id in data_list:
            data = {}
            data['video_id'] = video_id

            data['actions_present'] = self.get_actions_present(self.data_anno[video_id])

            data['final_score'] = self.data_anno[video_id]["final_score"]

            data['difficulty'] = self.data_anno[video_id]["difficulty"]

            data['gt_score'] = data['final_score']

            processed_data_list.append(data)
        

        return processed_data_list

    def get_actions_present(self, anno):
        
        """       
        armstand: No, Yes
        rotation_type: Inward, reverse, backward, forward
            
        positions: Free, tuck, Pike
        #SS: 9 unique , including 0 for no ss
        #tw: 8 unique, including 0 for no tw

        indexing: 0, 1,2,3,4, 5,6,7, 8,9,10,11,12,13,14,15,  16,17,18,19,20,21,22
        """
        if anno["armstand"] != 0:
            armstand_idx = 0
        else:
            armstand_idx = "MISSING"

        rotation_type_idx = 1 + anno["rotation_type"]

        pos_idx = 5 + anno["position"]
            
        if anno["ss_no"] != 0:
            ss_idx = 7 + anno["ss_no"]
        else:
            ss_idx = "MISSING"

        if anno["tw_no"] != 0:
            tw_idx = 15 + anno["tw_no"]
        else:
            tw_idx = "MISSING"

        actions_present = []
        for idx in [pos_idx, armstand_idx, rotation_type_idx, ss_idx, tw_idx]:
            if idx != "MISSING":
                actions_present.append(idx)

        #action for entry
        actions_present.append(23)

        return actions_present

    def read_pickle(self, pickle_path):
        with open(pickle_path,'rb') as f:
            pickle_data = pickle.load(f)
        return pickle_data


    def load_video(self, video_file_name):
        first, second = video_file_name[0], video_file_name[1]

        if first < 10:
            first = "0"+str(first)
        
        if second < 10:
            second = "0"+str(second)
        image_list = sorted((glob.glob(os.path.join(self.video_dir, str(first)+"_"+str(second), '*.jpg'))))

        if self.is_training:
            #start from 0-1-2-3
            start_idx = torch.randint(0,4,[1]).item()
            image_list = image_list[start_idx:start_idx+self.frames_per_clip]

        video = [Image.open(image) for image in image_list]
        video = [transforms.ToTensor()(frame) for frame in video]
        video = torch.stack(video)
        video = self.transforms(video)

        #extract windows
        start_idx = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

        video_pack = torch.stack([video[i:i+self.window_size,:,:,:] for i in start_idx])

        return video_pack

    def load_feats(self, video_file_name):
        feats = np.load(os.path.join(self.feat_dir, str(video_file_name[0])+"-abr-"+str(video_file_name[1])+'.npz'))["arr_0"]
        feats = torch.from_numpy(feats).float()
            
        return feats


    def __getitem__(self, index):
        video_data = self.datalist[index]

        video_id = video_data["video_id"]

        data = {"video_id": video_id}
        if self.use_feats:
            data['feats'] = self.load_feats(video_id)
        else:
            data['window_frames'] = self.load_video(video_id)
        
        data["video_name"] = video_id
        data["difficulty"] = video_data["difficulty"]
        data["actions_present"] = video_data["actions_present"]     

        target = video_data["gt_score"]
        data["gt_score"] =  video_data["gt_score"]

        if self.with_dd:
            target = target / data['difficulty']

        if self.three_judge_score_scaling:
            target = target / 3.0

        data["target"] = target
        
        return data


    def __len__(self):
        return len(self.datalist)
