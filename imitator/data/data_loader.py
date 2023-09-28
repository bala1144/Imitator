import os
import torch
from collections import defaultdict
from torch.utils import data
import copy
import numpy as np
import pickle
from tqdm import tqdm
import random,math
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor
import librosa    
import pytorch_lightning as pl

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data,subjects_dict,data_type="train"):
        self.data = data
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type
        self.one_hot_labels = np.eye(len(subjects_dict["train"]))

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        file_name = self.data[index]["name"]
        audio = self.data[index]["audio"]
        vertice = self.data[index]["vertice"]
        template = self.data[index]["template"]
        if self.data_type == "train":
            subject = "_".join(file_name.split("_")[:-1])
            one_hot = self.one_hot_labels[self.subjects_dict["train"].index(subject)]
        else:
            one_hot = self.one_hot_labels
        return torch.FloatTensor(audio),torch.FloatTensor(vertice), torch.FloatTensor(template), torch.FloatTensor(one_hot), file_name

    def __len__(self):
        return self.len

def read_data(
        dataset,
        dataset_root,
        wav_path,
        vertices_path,
        template_file,
        train_subjects,
        val_subjects,
        test_subjects
              ):
    print("Loading data...")
    data = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []

    audio_path = os.path.join(os.getenv("HOME"), dataset_root, wav_path)
    vertices_path = os.path.join(os.getenv("HOME"), dataset_root, vertices_path)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    template_file = os.path.join(os.getenv("HOME"), dataset_root, template_file)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin,encoding='latin1')
    
    for r, ds, fs in os.walk(audio_path):
        for f in tqdm(fs):
            if f.endswith("wav"):
                wav_path = os.path.join(r,f)
                speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
                input_values = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
                key = f.replace("wav", "npy")
                data[key]["audio"] = input_values
                subject_id = "_".join(key.split("_")[:-1])
                temp = templates[subject_id]
                data[key]["name"] = f
                data[key]["template"] = temp.reshape((-1)) 
                vertice_path = os.path.join(vertices_path,f.replace("wav", "npy"))
                if not os.path.exists(vertice_path):
                    del data[key]
                else:
                    if dataset == "vocaset":
                        data[key]["vertice"] = np.load(vertice_path,allow_pickle=True)[::2,:]#due to the memory limit
                    elif dataset == "BIWI":
                        data[key]["vertice"] = np.load(vertice_path,allow_pickle=True)

    subjects_dict = {}
    subjects_dict["train"] = [i for i in train_subjects.split(" ")]
    subjects_dict["val"] = [i for i in val_subjects.split(" ")]
    subjects_dict["test"] = [i for i in test_subjects.split(" ")]

    splits = {'vocaset':{'train':range(1,41),'val':range(21,41),'test':range(21,41)},
     'BIWI':{'train':range(1,33),'val':range(33,37),'test':range(37,41)}}
   
    for k, v in data.items():
        subject_id = "_".join(k.split("_")[:-1])
        sentence_id = int(k.split(".")[0][-2:])
        if subject_id in subjects_dict["train"] and sentence_id in splits[dataset]['train']:
            train_data.append(v)
        if subject_id in subjects_dict["val"] and sentence_id in splits[dataset]['val']:
            valid_data.append(v)
        if subject_id in subjects_dict["test"] and sentence_id in splits[dataset]['test']:
            test_data.append(v)

    print(len(train_data), len(valid_data), len(test_data))
    return train_data, valid_data, test_data, subjects_dict

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        train_data, valid_data, test_data, subjects_dict = read_data(**kwargs)
        self.train_data = Dataset(train_data, subjects_dict, "train")
        self.valid_data = Dataset(valid_data, subjects_dict, "val")
        self.test_data = Dataset(test_data, subjects_dict, "test")

        # need for the pytorhch lightning modeule to work
        self.dataset_configs = {}
        self.dataset_configs["train"] = self.train_data
        self.train_dataloader = self._train_dataloader
        self.dataset_configs["validation"] = self.valid_data
        self.val_dataloader = self._val_dataloader
        self.dataset_configs["test"] = self.test_data
        self.test_dataloader = self._test_dataloader

        self.data_cfg = kwargs

    def prepare_data(self):
        print("prepere data do nothin")

    def setup(self, stage=None):
        print("setup do nothin")

    def _train_dataloader(self):
        return data.DataLoader(dataset=self.train_data, batch_size=1, shuffle=True)

    def _val_dataloader(self):
        return data.DataLoader(dataset=self.valid_data, batch_size=1, shuffle=True)

    def _test_dataloader(self):
        return data.DataLoader(dataset=self.test_data, batch_size=1, shuffle=True)

    def _test_unseen_dataloader(self):
        # Need to this place holder function for the tester
        return None
