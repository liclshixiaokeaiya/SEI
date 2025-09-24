import json
import torch
import os
from torch.utils.data import Dataset
from PIL import Image


class SEIDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 json_path: str,
                 transform=None,
                 indices=None,
                 use_threshold_samples=False,
                 threshold_samples_set_idx=1,
                 is_test=True, 
                 seed=1,
                 tokenizer=None):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.is_test = is_test
        self.seed = seed
        self.tokenize = tokenizer

        with open(json_path, 'r') as f:
            records = json.load(f)
        
        self.samples = [rec["Name"] for rec in records]
        self.true_labels = torch.tensor([rec["True_Label"] for rec in records], dtype=torch.long)
        self.noisy_labels = torch.tensor([rec["After_Label"] for rec in records], dtype=torch.long)

        N = len(self.samples)
        self.indices = torch.arange(N) if indices is None else torch.tensor(indices, dtype=torch.long)

        
        self.flip_dict = {int(i): int(self.noisy_labels[i]) for i in self.indices.tolist()}

       
        self.use_threshold_samples = use_threshold_samples
        
        if use_threshold_samples:
            torch.manual_seed(self.seed)
            
            K = int(self.true_labels.max().item()) + 1
            M = len(self.indices) // K
            start = (threshold_samples_set_idx - 1) * M
            end   = threshold_samples_set_idx * M
            perm = torch.randperm(len(self.indices))
            self.threshold_sample_indices = perm[start:end]

    @property
    def targets(self) -> torch.Tensor:
        return self.true_labels[self.indices]

    @property
    def assigned_targets(self) -> torch.Tensor:
        
        assigned = self.targets.clone()
        for i, idx in enumerate(self.indices.tolist()):
            assigned[i] = self.flip_dict[idx]
        if self.use_threshold_samples:
            extra = int(self.targets.max().item()) + 1
            assigned[self.threshold_sample_indices] = extra
        return assigned

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        global_idx = int(self.indices[index].item())
        img_path = os.path.join(self.data_dir, self.samples[global_idx])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = int(self.assigned_targets[index].item())
        if self.is_test:
            truelabel = int(self.true_labels[global_idx].item())
            return img, truelabel, label, index
        else:
            return img, label, index