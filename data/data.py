import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

class data_loader:
    def __init__(self, file_path, max_length):
        self.file_path = file_path
        self.load_assemble_data()
        self.aa_table = self._init_aa_table()
        
        self.feature_coding = self.features_to_indexSequence(max_length = max_length)
    def _init_aa_table(self):
        aa_table = {}
        aa_table['A'] = 1
        aa_table['C'] = 2
        aa_table['D'] = 3
        aa_table['E'] = 4
        aa_table['F'] = 5
        aa_table['G'] = 6
        aa_table['H'] = 7
        aa_table['I'] = 8
        aa_table['K'] = 9
        aa_table['L'] = 10
        aa_table['M'] = 11
        aa_table['N'] = 12
        aa_table['P'] = 13
        aa_table['Q'] = 14
        aa_table['R'] = 15
        aa_table['S'] = 16
        aa_table['T'] = 17
        aa_table['V'] = 18
        aa_table['W'] = 19
        aa_table['Y'] = 20
        return aa_table
    def load_assemble_data(self):
        data = np.loadtxt(self.file_path, delimiter = ",", skiprows = 1, dtype = str)
        features = data[:,0]
        labels = data[:,1].astype(float)
        self.features = features
        self.labels = np.array(labels)
        print(f"Info: data loading completely, {len(self.features)} contained")
    
    def features_to_indexSequence(self, max_length):
        result = []
        for str_sequence in self.features:
            idx_sequence = [self.aa_table[x] for x in str_sequence]

            padding = max_length - len(idx_sequence)
            if padding > 0:
                idx_sequence += [0] * padding

            result.append(idx_sequence)
        return np.array(result)

    def get_dataset(self, batch_size):
        t_label, v_label, t_feature, v_feature = \
        train_test_split(self.labels, self.feature_coding, test_size=0.3, random_state=42, shuffle=True)

        t_label = torch.from_numpy(t_label).float()
        v_label = torch.from_numpy(v_label).float()
        t_feature = torch.from_numpy(t_feature).long()
        v_feature = torch.from_numpy(v_feature).long()
        print(f"Train: {type(t_feature)} {t_feature.shape}")
        print(f"Valid: {type(v_feature)} {v_feature.shape}")

        t_dataset = TensorDataset(t_feature, t_label)
        v_dataset = TensorDataset(v_feature, v_label)

        self.t_loader = DataLoader(t_dataset, batch_size = batch_size, shuffle = True)
        self.v_loader = DataLoader(v_dataset, batch_size = batch_size, shuffle = True)