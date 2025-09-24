import math
import os
from collections import defaultdict, namedtuple
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import torch
import torch.nn.functional as F

sample_identifier = Union[int, str]
eps = 1e-8  


@dataclass
class RecordWithList:
    sample_id: sample_identifier
    num_measurements: int
    target_logit: int
    target_val: float
    other_logit: int
    other_val: float
    margin: float
    sei: float
    class_values: List[float] = field(default_factory=list)
            
    

class MyCalculatorEntropySigned():
    def __init__(self, save_dir: str, compressed: bool = True):
       
        self.save_dir = save_dir
        self.counts = defaultdict(int)
        self.sums = defaultdict(float)

        self.compressed = compressed
        if not compressed:
            self.records = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor,
               sample_ids: List[sample_identifier]):
        
        target_values = logits.gather(1, targets.view(-1, 1)).squeeze()

        masked_logits = torch.scatter(logits, 1, targets.view(-1, 1), float('-inf'))
        other_logit_values, other_logit_index = masked_logits.max(1)
        other_logit_values = other_logit_values.squeeze()
        other_logit_index = other_logit_index.squeeze()
        
        margin_values = target_values - other_logit_values

        shannon_entropy_values = -torch.sum(logits * torch.log(logits + eps), dim=1)
        
        signs = torch.ones_like(margin_values)
        signs[margin_values < 0] = -1.0
        
        signed_entropy = signs * shannon_entropy_values
        
        margin_entropy_signed_values = signed_entropy.tolist()

        updated_seis = {}
        for i, (sample_id, margin, logit) in enumerate(zip(sample_ids, margin_entropy_signed_values, logits)):
            self.counts[sample_id] += 1
            self.sums[sample_id] += margin
            
            record = RecordWithList(sample_id=sample_id,
                               num_measurements=self.counts[sample_id],
                               target_logit=targets[i].item(),
                               target_val=target_values[i].item(),
                               other_logit=other_logit_index[i].item(),
                               other_val=other_logit_values[i].item(),
                               margin=margin,
                               sei=self.sums[sample_id] / self.counts[sample_id],
                               class_values=logit.tolist(),
                               )

            updated_seis[sample_id] = record
            if not self.compressed:
                self.records.append(record)

        return updated_seis
    

    def finalize(self, save_dir: Optional[str] = None) -> None:
        
        save_dir = save_dir or self.save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        results = [{
            'sample_id': sample_id,
            'sei': self.sums[sample_id] / self.counts[sample_id]
        } for sample_id in self.counts.keys()]

        result_df = pd.DataFrame(results).sort_values(by='sei', ascending=False)

        save_path = os.path.join(save_dir, 'sei_values.csv')
        result_df.to_csv(save_path, index=False)

        if not self.compressed:
            records_df = MyCalculatorEntropySigned.records_to_df(self.records)
            save_path = os.path.join(save_dir, 'full_sei_records.csv')
            records_df.to_csv(save_path, index=False)

    
    @staticmethod
    def records_to_df(records: List['RecordWithList']) -> pd.DataFrame:
       
        if not records:
            return pd.DataFrame()

        df = pd.DataFrame([asdict(record) for record in records])

        
        logits_df = df['class_values'].apply(pd.Series)
        logits_df.columns = [f'class_{i}_logit' for i in range(logits_df.shape[1])]


        df = pd.concat([df, logits_df], axis=1)

        df.drop(columns=['class_values'], inplace=True)

        df.sort_values(by=['sample_id', 'num_measurements'], inplace=True)
        return df
    
    

