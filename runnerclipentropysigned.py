import datetime
import logging
import os
import random
import sys
from collections import OrderedDict

import pandas as pd
import torch

import fire
import tqdm
import util
from EntropySignedCalculator import MyCalculatorEntropySigned
from MySEIdataset import SEIDataset
from MyClipLoss import ClipLoss
from torchvision import transforms
import torch.nn as nn
from torch.nn import functional as F
from open_clip import create_model
import open_clip

import sys


try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def build_class_tokens_for_dataset(dataset: str):
    
    ds = (dataset or "").strip().lower()
    if ds == "isic":
        class_names = [
            'melanoma',
            'nevus',
            'basal cell carcinoma',
            'actinic keratosis/intraepithelial carcinoma',
            'benign keratosis',
            'dermatofibroma',
            'vascular lesion',
            'other lesions'
        ]
        prompt_tpl = "A dermoscopic image showing {name}."
    elif ds == "drid":
        class_names = [
            'showing no evidence of diabetic retinopathy',
            'exhibiting mild diabetic retinopathy',
            'exhibiting moderate diabetic retinopathy',
            'exhibiting severe diabetic retinopathy',
            'exhibiting proliferative diabetic retinopathy',
            'other retinal conditions'
        ]
        prompt_tpl = "A fundus image {name}."
    elif ds == "panda":
        class_names = [
            'benign glandular epithelium',
            'Gleason pattern 3 adenocarcinoma',
            'Gleason pattern 4 adenocarcinoma',
            'Gleason pattern 5 adenocarcinoma',
            'other conditions'
        ]
        prompt_tpl = "A histology image showing {name}."
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Supported: isic | drid | panda")

    tokenizer = open_clip.get_tokenizer('RN50')
    class_prompts = [prompt_tpl.format(name=name) for name in class_names]
    class_tokens = tokenizer(class_prompts).cuda() 
    print("class_names:", class_names)
    return class_names, tokenizer, class_prompts, class_tokens


class RunnerCLIPEntropySigned(object):

    def __init__(self,
                 data,
                 save,
                 jsondir,
                 dataset="isic",
                 num_valid=5000,
                 seed=0,
                 split_seed=None,
                 noise_type="uniform",
                 perc_mislabeled=0.,
                 use_threshold_samples=False,
                 threshold_samples_set_idx=1,
                 loss_type="cross-entropy",
                 oracle_training=False,
                 net_type="resnet",
                 pretrained=False,
                 num_classes=7,
                 **model_args):
        if not os.path.exists(save):
            os.makedirs(save)
        if not os.path.isdir(save):
            raise Exception('%s is not a dir' % save)
        self.data = data
        self.jsondir = jsondir
        self.savedir = save
        self.perc_mislabeled = perc_mislabeled
        self.noise_type = noise_type
        self.dataset = dataset
        self.net_type = net_type
        self.num_valid = num_valid
        self.use_threshold_samples = use_threshold_samples
        self.threshold_samples_set_idx = threshold_samples_set_idx
        self.split_seed = split_seed if split_seed is not None else seed
        self.seed = seed
        self.oracle_training = oracle_training
        self.pretrained = pretrained

        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        random.seed(0)

        self.timestring = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        logging.basicConfig(
            format='%(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(os.path.join(self.savedir, 'log-%s.log' % self.timestring)),
            ],
            level=logging.INFO,
        )
        logging.info('Data dir:\t%s' % data)
        logging.info('Save dir:\t%s\n' % save)

        self.class_names, self.tokenizer, self.class_prompts, self.class_tokens = \
            build_class_tokens_for_dataset(self.dataset)

        
        self.num_classes = len(self.class_names)
        if use_threshold_samples:
            self.num_classes += 1
            print('self.num_classes:', self.num_classes)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(227 if "inception" in self.net_type else 224),
            transforms.ToTensor(),
            normalize,
        ])
        train_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(227 if "inception" in self.net_type else 224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        self.test_transforms = test_transforms
        self.train_transforms = train_transforms

        logging.info(f"Model type: {self.net_type}")
        logging.info(f"Model args:")
        for key, val in model_args.items():
            logging.info(f" - {key}: {val}")
        logging.info(f"Loss type: {loss_type}")
        logging.info("")

    def generate_sei_details(self, load=None):

        load = load or self.savedir
        train_data = torch.load(os.path.join(load, "train_data.pth"))
        sei_data = pd.read_csv(os.path.join(load, "sei_values.csv"))

        if "assigned_targets" not in train_data:
            train_data["assigned_targets"] = train_data["observed_targets"]

        true_targets = train_data["true_targets"]
        assigned_targets = train_data["assigned_targets"]
        is_threshold_sample = assigned_targets.gt(true_targets.max())
        label_flipped = torch.ne(true_targets, assigned_targets)

        result = {}

        result["Index"] = torch.arange(train_data["assigned_targets"].size(-1))

        result["True Target"] = true_targets
        result["Observed Target"] = assigned_targets
        result["Label Flipped"] = label_flipped
        result["Is Threshold Sample"] = is_threshold_sample

        sei_data = sei_data.set_index('sample_id')
        sei_data = sei_data.reindex(list(range(train_data["assigned_targets"].size(-1))))
        sei_list = sei_data['sei'].to_list()
        result["SEI"] = torch.tensor(sei_list)

        if is_threshold_sample.sum().item():
            sei_wtr = torch.lt(
                result["SEI"].view(-1, 1),
                result["SEI"][is_threshold_sample].view(1, -1),
            ).float().mean(dim=-1).gt(0.01).float()
            result["SEI_WTR"] = sei_wtr
        else:
            result["SEI_WTR"] = torch.ones_like(result["SEI"])

        df = pd.DataFrame(result)
        df.set_index(
            ["Index", "True Target", "Observed Target", "Label Flipped", "Is Threshold Sample"],
            inplace=True)
        df.to_csv(os.path.join(load, "sei_details.csv"))
        return self

    def done(self):
        "Break out of the runner"
        return None

    def load(self, save=None, suffix=""):

        save = save or self.savedir
        state_dict = torch.load(os.path.join(save, f"model.pth{suffix}"),
                                map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict, strict=False)
        return self

    def save(self, save=None, suffix=""):

        save = save or self.savedir
        torch.save(self.model.state_dict(), os.path.join(save, f"model.pth{suffix}"))
        return self

    def subset(self, perc, sei_files=None):

        if sei_files is None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            random.seed(self.seed)
            order = torch.randperm(len(self.train_set))
        else:
            counts = torch.zeros(len(self.train_set))
            seis = torch.zeros(len(self.train_set))
            if isinstance(sei_files, str):
                sei_files = sei_files.split(",")
            for sub_sei_file in sei_files:
                seis_path = os.path.join(sub_sei_file, "sei_details.csv")
                if not os.path.exists(seis_path):
                    self.compute_seis(load=sub_sei_file)
                seis_data = pd.read_csv(seis_path).drop(
                    ["True Target", "Observed Target", "Label Flipped"], axis=1)
                counts += torch.tensor(~seis_data["Is Threshold Sample"].values).float()
                seis += torch.tensor(seis_data["SEI"].values *
                                     ~seis_data["Is Threshold Sample"].values).float()
            counts.clamp_min_(1)
            seis = seis.div_(counts)
            order = seis.argsort(descending=True)

        num_samples = int(len(self.train_set) * perc)
        self.train_set.indices = self.train_set.indices[order[:num_samples]]
        logging.info(f"Reducing training set from {len(order)} to {len(self.train_set)}")
        if sei_files is not None:
            logging.info(
                f"Average SEI: {seis[order[:num_samples]].mean().item()} (from {seis.mean().item()}"
            )
        return self

    def train_for_sei_computation(self,
                                  num_epochs=150,
                                  batch_size=64,
                                  lr=0.1,
                                  wd=1e-4,
                                  momentum=0.9,
                                  **kwargs):

        return self.train(num_epochs=num_epochs,
                          batch_size=batch_size,
                          test_at_end=False,
                          lr=lr,
                          wd=wd,
                          momentum=momentum,
                          lr_drops=[],
                          **kwargs)

    def _make_model(self):
        model = create_model(
            model_name="RN50",
            pretrained="openai",
            device='cuda',
            output_dict=True,
        )
        return model

    def train(self,
              num_epochs=150,
              batch_size=256,
              lr=0.1,
              wd=1e-4,
              momentum=0.9,
              lr_drops=[0.5, 0.75],
              sei_wtr=False,
              rand_weight=False,
              **kwargs):

        self.model = self._make_model()

        self.loss = ClipLoss()

        optimizer = torch.optim.SGD(self.model.parameters(),
                                    lr=lr,
                                    weight_decay=wd,
                                    momentum=momentum,
                                    nesterov=True)
        milestones = [int(lr_drop * num_epochs) for lr_drop in (lr_drops or [])]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=milestones,
                                                         gamma=0.1)
        logging.info(f"\nOPTIMIZER:\n{optimizer}")
        logging.info(f"SCHEDULER:\n{scheduler.milestones}")
        print('json dir')
        print(self.jsondir)

        sei_calculator = MyCalculatorEntropySigned(save_dir=self.savedir, compressed=False)
        self.train_set = SEIDataset(data_dir=os.path.join(self.data),
                                    json_path=self.jsondir,
                                    transform=self.train_transforms,
                                    use_threshold_samples=self.use_threshold_samples,
                                    threshold_samples_set_idx=self.threshold_samples_set_idx,
                                    is_test=False,
                                    tokenizer=self.tokenizer)  

        train_data = OrderedDict()
        train_data["train_indices"] = self.train_set.indices

        train_data["true_targets"] = self.train_set.targets
        train_data["assigned_targets"] = self.train_set.assigned_targets

        results = []
        with torch.no_grad():
            self.text_features_all = self.model.encode_text(self.class_tokens)

        for epoch in range(num_epochs):
            train_results = self.train_epoch(model=self.model,
                                             optimizer=optimizer,
                                             epoch=epoch,
                                             num_epochs=num_epochs,
                                             batch_size=batch_size,
                                             sei_calculator=sei_calculator,
                                             sei_wtr=sei_wtr,
                                             rand_weight=rand_weight,
                                             **kwargs)

            logging.info(f"\nTraining {repr(train_results)}")
            logging.info('')
            results.append(
                OrderedDict([("epoch", f"{epoch + 1:03d}"),
                             *[(f"train_{field}", val) for field, val in train_results.items()],
                             ]))
            pd.DataFrame(results).set_index("epoch").to_csv(
                os.path.join(self.savedir, "train_log.csv"))

            torch.save(train_data, os.path.join(self.savedir, "train_data.pth"))

            sei_calculator.finalize()

            scheduler.step()

        sei_calculator.finalize()

        self.load(suffix=".last")

        return self

    def train_epoch(self,
                    model,
                    optimizer,
                    epoch,
                    num_epochs,
                    batch_size=256,
                    num_workers=0,
                    sei_calculator=None,
                    sei_wtr=False,
                    rand_weight=False):
        stats = ["loss"]
        meters = [util.AverageMeter() for _ in stats]
        result_class = util.result_class(stats)

        if sei_wtr:
            counts = torch.zeros(len(self.train_set))
            bad_probs = torch.zeros(len(self.train_set))
            if isinstance(sei_wtr, str):
                sei_wtr = sei_wtr.split(",")
            for sub_sei_wtr in sei_wtr:
                seis_path = os.path.join(sub_sei_wtr, "sei_details.csv")
                if not os.path.exists(seis_path):
                    self.generate_sei_details(load=sub_sei_wtr)
                seis_data = pd.read_csv(seis_path).drop(
                    ["True Target", "Observed Target", "Label Flipped"], axis=1)
                counts += torch.tensor(~seis_data["Is Threshold Sample"].values).float()
                bad_probs += torch.tensor(seis_data["SEI_WTR"].values *
                                          ~seis_data["Is Threshold Sample"].values).float()
            counts.clamp_min_(1)
            good_probs = (1 - bad_probs / counts).to(next(model.parameters()).dtype).ceil()
            if torch.cuda.is_available():
                good_probs = good_probs.cuda()
            logging.info(f"SEI WTR Score")
            logging.info(f"(Num samples removed: {good_probs.ne(1.).sum().item()})")
        elif rand_weight:
            logging.info("Rectified Normal Random Weighting")
        else:
            logging.info("Standard weighting")

        loader = tqdm.tqdm(torch.utils.data.DataLoader(self.train_set,
                                                       batch_size=batch_size,
                                                       shuffle=True,
                                                       num_workers=num_workers),
                           desc=f"Train (Epoch {epoch + 1}/{num_epochs})")

        model.train()
        for inputs, targets, indices in loader:
            optimizer.zero_grad()

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()

            image_features = model.encode_image(inputs)

            text_features = self.text_features_all

            lm = model.module if hasattr(model, "module") else model
            logit_scale_param = lm.logit_scale

            losses, logits_per_image = self.loss(image_features=image_features,
                                                text_features=text_features,
                                                logit_scale=logit_scale_param,
                                                gt_labels=targets,
                                                output_dict=True)
            total_loss = sum(losses.values())
            losses["loss"] = total_loss
            preds = logits_per_image.argmax(dim=1)
            logits_per_image_soft = F.softmax(logits_per_image, dim=1)

            sei_calculator.update(logits_per_image_soft.detach().cpu().half().float(),
                                  targets.detach().cpu(), indices.tolist())

            total_loss.backward()

            optimizer.step()

            batch_size = preds.size(0)
            stat_vals = [total_loss.item()]
            for stat_val, meter in zip(stat_vals, meters):
                meter.update(stat_val, batch_size)

            res = dict(
                (name, f"{meter.val:.3f} ({meter.avg:.3f})") for name, meter in zip(stats, meters))
            loader.set_postfix(**res)

        return result_class(*[meter.avg for meter in meters])


if __name__ == "__main__":
    fire.Fire(RunnerCLIPEntropySigned)
