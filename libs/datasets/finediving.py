import os
import json
import glob
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.nn import functional as F

from .datasets import register_dataset

import pickle


@register_dataset("finediving")
class FineDiving(Dataset):
    def __init__(self,
                is_training,
                split,
                data_root,
                video_dir,
                label_file,
                coarse_label_file,
                train_datafile,
                test_datafile,
                stride,
                window_size,
                frames_per_clip,
                max_seq_len,
                input_frame_size = None,
                crop_frame_size = 224,
                with_dd = True,
                three_judge_score_scaling = False, 
                use_feats = None,
                feat_dir = None
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

        self.label_path = os.path.join(self.data_root,"Annotations", label_file)
        self.coarse_label_path = os.path.join(self.data_root,"Annotations", coarse_label_file)

        train_datafile_path = os.path.join(self.data_root, "Annotations",train_datafile)
        test_datafile_path = os.path.join(self.data_root,"Annotations", test_datafile)

        #mapping from subaction to index, refer to gen_text_embeddings.py for more details
        self.label_to_idx_dict = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 37: 7, 12: 8, 13: 9, 14: 10, 15: 11, 
                                16: 12, 17: 13, 19: 14, 21: 15, 22: 16, 23: 17, 24: 18, 25: 19, 27: 20, 29: 21, 
                                30: 22, 31: 23, 32: 24, 33: 25, 34: 26, 35: 27, 36: 28}

        
        self.data_anno = self.read_pickle(self.label_path)
        self.coarse_data_anno = self.read_pickle(self.coarse_label_path)
        
        if self.subset == 'test':
            self.datalist = self._load_annotations(test_datafile_path)
        elif self.subset == 'train':
            self.datalist = self._load_annotations(train_datafile_path)
        else:
            raise ValueError("subset should be train or test")          
        

    def _load_annotations(self, datafile_path):
        data_list = self.read_pickle(datafile_path)
        
        processed_data_list = []

        for video_id in data_list:
            data = {}
            data['video_id'] = video_id

            data['final_score'] = self.data_anno[video_id]["dive_score"]

            data['difficulty'] = self.data_anno[video_id]["difficulty"]

            data['gt_score'] = data['final_score']

            processed_data_list.append(data)

        return processed_data_list

    def load_video(self, video_file_name):
        image_list = sorted((glob.glob(os.path.join(self.video_dir, video_file_name[0], str(video_file_name[1]), '*.jpg'))))

        if self.is_training:
            #start from 0-1-2-3
            start_idx = torch.randint(0,4,[1]).item()

            # randomly drop the end frames 
            end_idx = torch.randint(0,4,[1]).item()
            image_list = image_list[start_idx:start_idx + len(image_list) - end_idx]

        start_frame = int(image_list[0].split("/")[-1][:-4])
        end_frame = int(image_list[-1].split("/")[-1][:-4])

        frame_list = np.linspace(start_frame, end_frame, self.frames_per_clip).astype(np.int32)
        image_frame_idx = [frame_list[i] - start_frame for i in range(self.frames_per_clip)]
        video = [Image.open(image_list[image_frame_idx[i]]) for i in range(self.frames_per_clip)]
        video = [transforms.ToTensor()(frame) for frame in video]
        video = torch.stack(video)
        video = self.transforms(video)

        #extract windows
        start_idx = list(range(0, 90, 10))

        video_pack = torch.stack([video[i:i+self.window_size,:,:,:] for i in start_idx])

        frames_labels = self.data_anno[video_file_name]["frames_labels"]
        #mapping labels to label indices
        frames_labels = [self.label_to_idx_dict[l] for l in frames_labels]

        return video_pack, frames_labels



    def load_feats(self, video_file_name):
        feats = np.load(os.path.join(self.feat_dir, video_file_name[0]+"-abr-"+str(video_file_name[1])+'.npz'))["feats"]

        feats = feats.transpose(1,0)

        feats = torch.from_numpy(feats).float()

        frames_labels = self.data_anno[video_file_name]["frames_labels"]

        #mapping labels to label indices
        frames_labels = [self.label_to_idx_dict[l] for l in frames_labels]

        return feats, frames_labels


    def read_pickle(self, pickle_path):
        with open(pickle_path,'rb') as f:
            pickle_data = pickle.load(f)
        return pickle_data

    
    def __getitem__(self, index):
        video_data = self.datalist[index]

        video_id = video_data["video_id"]

        data = {"video_id": video_id}
        if self.use_feats:
            data['feats'], frame_labels = self.load_feats(video_id)
        else:
            data['window_frames'], frame_labels = self.load_video(video_id)


        data["video_name"] = video_id
        
        data['difficulty'] = video_data["difficulty"]
        
        target = video_data["gt_score"]
        data["gt_score"] = video_data["gt_score"]

        if self.with_dd:
            target = target / data['difficulty']

        if self.three_judge_score_scaling:
            target = target / 3.0
        
        data["target"] =  target

        #if entry is missing, add 28
        if 28 not in frame_labels:
            frame_labels.append(28)

        #take only the presence of actions
        data['actions_present'] = sorted(list(set(frame_labels)))

        return data

    def __len__(self):
        return len(self.datalist)
