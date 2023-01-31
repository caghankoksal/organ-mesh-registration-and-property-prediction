import os 
import torch
import pandas as pd 
import numpy as np
from torch_geometric.data import Dataset, download_url, Data
import open3d as o3d
from sklearn.preprocessing import MinMaxScaler

class OrganMeshDataset(Dataset):
    def __init__(self, config, mode='train', transform=None, pre_transform=None, pre_filter=None,
                  pre_process=True):
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
    
        super().__init__(config, transform, pre_transform, pre_filter)
        assert mode in ['train', 'val', 'test']
        assert config.organ in ['left_kidney', 'liver', 'pancreas', 'right_kidney', 'spleen']

        self.root = config.root
        self.organ = config.organ
        self.pre_process = pre_process
        self.task = config.task
        self.use_registered_data = config.use_registered_data
        self.decimation_path = config.decimation_path
        self.registeration_path = config.registeration_path
        self.use_scaled_age = config.use_scaled_age
  
        split_path = os.path.join(config.split_path, f'organs_split_{mode}.txt')
        with open(split_path) as f:
            self.organ_mesh_ids = f.readlines()

        self.organ_mesh_ids = [each.replace('\n','') for each in self.organ_mesh_ids]

        # Select number of samples according to the mode
        num_samples = config.num_train_samples if mode == 'train' else config.num_test_samples
        if num_samples is not None:
            self.organ_mesh_ids = self.organ_mesh_ids[:num_samples]    

         
        self.bridge_path = config.bridge_path

        self.basic_features = pd.read_csv(config.basic_feat_path)
        new_names = {'21003-2.0':'age', '31-0.0':'sex', '21001-2.0':'bmi', '21002-2.0':'weight','50-2.0':'standing_weight'}
        self.basic_features = self.basic_features.rename(index=str, columns=new_names)

        # Scale Age
        if config.use_scaled_age:
            # Scale the age using MinMaxScaler() from sklearn
            scaler = MinMaxScaler()
            self.basic_features['age'] = scaler.fit_transform(self.basic_features['age'].values.reshape(-1,1))

        self.bridge_organ_df = pd.read_csv(config.bridge_path)
    
        if self.pre_process:
            self.patient_feats = {}
            for cur_patient in self.organ_mesh_ids:
                cur_patient_features = self.basic_features[self.basic_features['eid'] == int(cur_patient)]
                self.patient_feats[cur_patient] = cur_patient_features
    
        print(f'{self.organ.capitalize()}  {mode} Dataset is created')
    def len(self):
        return len(self.organ_mesh_ids)

    def get(self, idx):
        selected_patient = self.organ_mesh_ids[idx]
        #print('Selected Patient', selected_patient)
        if self.use_registered_data:
            registered_mesh = []
            organ = f'{self.organ}_mesh.ply' 
            vertices_data = o3d.io.read_point_cloud(os.path.join(self.registeration_path, str(selected_patient), organ))
            edges_data = o3d.io.read_triangle_mesh(os.path.join(self.decimation_path, str(selected_patient), organ))
            vertices = torch.from_numpy(np.asarray(vertices_data.points)).double()
            triangles = np.asarray(edges_data.triangles)

            edges = []
            for triangle in triangles:
                edges.append([triangle[0], triangle[1]])
                edges.append([triangle[0], triangle[2]])
                edges.append([triangle[1], triangle[2]])

            edges_torch = torch.from_numpy(np.unique(np.array(edges), axis=0).reshape(2,-1)).long()
            eid = self.bridge_organ_df[self.bridge_organ_df['eid_87802'] == int(selected_patient)]["eid_60520"].values[0]
            sex = 0 if int(self.basic_features["sex"][self.basic_features.index[self.basic_features['eid'] == int(selected_patient)]]) == 0 else 1
            registered_mesh.append((vertices.type(torch.float32), edges_torch, sex, str(eid)))
            data = Data(x=registered_mesh[0][0], edge_index=registered_mesh[0][1], y=registered_mesh[0][2], num_nodes= len(registered_mesh[0][0]), eid=registered_mesh[0][3])
        else:
            data = torch.load(os.path.join(self.root, selected_patient,f'{self.organ}_mesh.pt'))
    
        #old_id = data['eid']
        #new_id = selected_patient
        # This might be bottleneck @TODO
        if self.pre_process:
            patient_features = self.patient_feats[selected_patient]
        else:
            patient_features = self.basic_features[self.basic_features['eid'] == int(selected_patient)]
        #print(patient_features['sex'])
        if self.task== 'sex_prediction':
            gender_patient = patient_features['sex'].item()
            #Label of the data is currently gender
            data.y = int(gender_patient)
        elif self.task == 'age_prediction':
            gender_age = patient_features['age'].item()
            if self.use_scaled_age:
                data.y =  gender_age
            else:
                data.y = int(gender_age)
        
        return data
    
