import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import os
import random

from utils import pad_random, pad


class ASVspoof2019(Dataset):
    def __init__(self, ids, dir_path, labels, pad_fn=pad_random, is_train=True):
        self.ids = ids
        self.labels = labels
        self.dir_path = dir_path
        self.cut = 64600
        self.is_train = is_train
        self.pad_fn = pad_fn

    def __getitem__(self, index):
        path_to_flac = f"{self.dir_path}/flac/{self.ids[index]}.flac"
        audio, rate = sf.read(path_to_flac)
        x_pad = self.pad_fn(audio, self.cut)
        x_inp = Tensor(x_pad)
        if not self.is_train:
            return x_inp, self.ids[index], torch.tensor(self.labels[index])
        return x_inp, torch.tensor(self.labels[index]), rate

    def __len__(self):
        return len(self.ids)

class EvalDataset(Dataset):
    def __init__(self, ids, dir_path, pad_fn=pad_random, cut=64600):
        self.ids = ids
        self.dir_path = dir_path
        self.cut = cut
        self.pad_fn = pad_fn

    def __getitem__(self, index):
        path_to_wav = f"{self.dir_path}/{self.ids[index]}"
        audio, rate = sf.read(path_to_wav)
        x_pad = self.pad_fn(audio, self.cut)
        x_inp = Tensor(x_pad)
        return x_inp, self.ids[index]

    def __len__(self):
        return len(self.ids)


class IndexSampler(torch.utils.data.sampler.Sampler[int]):
    """
    Класс для семплирования батчей с картинками индекса.

    :param data_source: Это датасет RTSD с синтетическими примерами
    :param examples_per_class: Число элементов каждого класса, которые должны попасть в индекс
    """

    def __init__(self, data_source: ASVspoof2019, examples_per_class: int) -> None:
        self.dataset = data_source
        self.examples_per_class = examples_per_class

    def __iter__(self):
        """
        Функция, которая будет генерировать список индексов элементов в батче.
        """
        batch = []
        # Какие классы?
        classes = list(self.dataset.classes_to_samples.keys())
        for cls in classes:
            elements = random.choices(self.dataset.classes_to_samples[cls], k=self.examples_per_class)
            batch.extend(elements)
            
        return iter(batch)

    def __len__(self) -> int:
        """
        Возвращает общее количество индексов.
        """
        # Что возвращать?
        n_classes = len(list(self.dataset.classes_to_samples.keys()))
        return len(self.dataset) // (n_classes * self.examples_per_class)


def get_data_for_evaldataset(path):
    ids_list = os.listdir(path)
    return ids_list


def get_data_for_dataset(path):
    ids_list = []
    label_list = []
    with open(path, "r") as file:
        for line in file:
            line = line.split()
            id, label = line[1], line[-1]
            ids_list.append(id)
            label = 1 if label == "bonafide" else 0
            label_list.append(label)
    return ids_list, label_list


def get_datasets(config):
    if config["model"] == "Res2TCNGuard":
        val_pad_fn = pad
    else:
        val_pad_fn = pad_random

    train_ids, train_labels = get_data_for_dataset(config["train_label_path"])
    train_dataset = ASVspoof2019(
        train_ids,
        config["train_path_flac"],
        train_labels
    )

    dev_ids, dev_labels = get_data_for_dataset(config["dev_label_path"])
    dev_dataset = ASVspoof2019(
        dev_ids,
        config["dev_path_flac"],
        dev_labels,
        val_pad_fn,
        False
    )

    eval_ids, eval_labels = get_data_for_dataset(config["eval_label_path"])

    eval_dataset = ASVspoof2019(eval_ids, config["eval_path_flac"], eval_labels, val_pad_fn, False)

    return {
        "train": train_dataset,
        "dev": dev_dataset,
        "eval": eval_dataset
    }


def get_dataloaders(datasets, config):
    dataloaders = {}

    if datasets.get("train"):
        train_loader = DataLoader(
            datasets["train"],
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"]
        )
        dataloaders["train"] = train_loader

    if datasets.get("dev"):
        dev_loader = DataLoader(
            datasets["dev"],
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"]
        )
        dataloaders["dev"] = dev_loader

    if datasets.get("train_knn"):
        sampler = IndexSampler(datasets["train_knn"], examples_per_class=1000)
        knn_train_loader = DataLoader(
            datasets["train_knn"],
            batch_sampler=[sampler],
            num_workers=config["num_workers"]
        )
        dataloaders["train_knn"] = knn_train_loader

    if datasets.get("eval"):
        eval_loader = DataLoader(
            datasets["eval"],
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"]
        )
        dataloaders["eval"] = eval_loader

    return dataloaders

