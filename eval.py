import argparse

import torch
from torch import nn
from utils import load_checkpoint, pad

from dataset import get_data_for_dataset, ASVspoof2019, get_dataloaders
from model.model import get_model
from metrics import produce_evaluation_file, evaluate_EER


def main(args, cfg):
    eval_ids, eval_labels = get_data_for_dataset(cfg['eval_label_path'])
    train_ids, train_labels = get_data_for_dataset(cfg['train_label_path'])

    eval_dataset = ASVspoof2019(eval_ids, cfg['eval_path_flac'], eval_labels, pad, False)
    eval_dataset = {
        "eval": eval_dataset
    }

    train_dataset = ASVspoof2019(train_ids, cfg['train_path_flac'], train_labels)
    train_dataset = {
        "train_knn": train_dataset
    }

    eval_dataloader = get_dataloaders(eval_dataset, cfg)
    train_knn_dataloader = get_dataloaders(train_dataset, cfg)

    model = get_model(cfg["checkpoint"], cfg["device"])

    model.train_head(train_knn_dataloader["train_knn"])
    model.save_head(cfg['knn_weights_path'])

    loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 0.9]).to(cfg["device"]))

    produce_evaluation_file(
        eval_dataloader["eval"],
        model,
        cfg["device"],
        loss_fn,
        cfg["produced_file"],
        cfg["eval_label_path"]
    )
    
    print(evaluate_EER(
        pred_df=cfg["produced_file"],
        ref_df=cfg["eval_label_path"],
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--config',
                        type=str,
                        default='configs/config.json')
    parser.add_argument('--output_file',
                        type=str,
                        default='eval_results.txt')
    args = parser.parse_args()
    config = load_checkpoint(args.config)
    main(args, config)
