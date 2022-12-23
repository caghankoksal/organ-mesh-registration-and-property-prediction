import os 
import torch
import pandas as pd 
from torch_geometric.data import Dataset, download_url


import os 
import torch
import pandas as pd 
from torch_geometric.data import Dataset, download_url


class OrganMeshDataset(Dataset):
    def __init__(self, root, basic_feats_path, bridge_path, split_path, mode='train', organ='liver', 
                 num_samples = None, transform=None, pre_transform=None, pre_filter=None, pre_process=True):
        """ Pytorch Geometric Organ Mesh Dataset 

        Args:
            root (str): Path to the mesh pt files
            basic_feats_path (str): path to the basic csv features file.
            bridge_path (str): path to the bridge file
            split_path (path): _description_. Defaults to None.
            mode (str, optional): split of the dataset. Defaults to 'train'.
            organ (str, optional): Organ. Defaults to 'liver'.
            num_samples (int, optional): to filter number of samples. Defaults to None.
            transform (_type_, optional): _description_. Defaults to None.
            pre_transform (_type_, optional): _description_. Defaults to None.
            pre_filter (_type_, optional): _description_. Defaults to None.
        """
    
        super().__init__(root, transform, pre_transform, pre_filter)
        assert mode in ['train', 'val', 'test']
        assert organ in ['left_kidney', 'liver', 'pancreas', 'right_kidney', 'spleen']

        self.root = root
        self.organ = organ
        self.pre_process = pre_process
        
        split_path = os.path.join(split_path, f'organs_split_{mode}.txt')
        with open(split_path) as f:
            self.organ_mesh_ids = f.readlines()

        self.organ_mesh_ids = [each.replace('\n','') for each in self.organ_mesh_ids]
        if num_samples is not None:
            self.organ_mesh_ids = os.listdir(root)[:num_samples]    

        self.basic_feats_path = basic_feats_path 
        self.bridge_path = bridge_path

        self.basic_features = pd.read_csv(basic_feats_path)
        new_names = {'21003-2.0':'age', '31-0.0':'sex', '21001-2.0':'bmi', '21002-2.0':'weight','50-2.0':'standing_weight'}
        self.basic_features = self.basic_features.rename(index=str, columns=new_names)
        self.bridge_organ_df = pd.read_csv(bridge_path)


        if self.pre_process:
            self.patient_feats = {}
            for cur_patient in self.organ_mesh_ids:
                cur_patient_features = self.basic_features[self.basic_features['eid'] == int(cur_patient)]
                self.patient_feats[cur_patient] =cur_patient_features
    
        print(f'{organ.capitalize()}  {mode} Dataset is created')
    def len(self):
        return len(self.organ_mesh_ids)

    def get(self, idx):
        selected_patient = self.organ_mesh_ids[idx]
        #print('Selected Patient', selected_patient)
        data = torch.load(os.path.join(self.root, selected_patient,f'{self.organ}_mesh.pt'))
        #old_id = data['eid']
        #new_id = selected_patient
        # This might be bottleneck @TODO
        if self.pre_process:
            patient_features = self.patient_feats[selected_patient]
        else:
            patient_features = self.basic_features[self.basic_features['eid'] == int(selected_patient)]
        #print(patient_features['sex'])
        gender_patient = patient_features['sex'].item()
        #Label of the data is currently gender
        data.y = int(gender_patient)
        return data
    
