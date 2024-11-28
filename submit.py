import argparse
from utils import load_checkpoint, pad

from dataset import get_data_for_evaldataset, EvalDataset, get_dataloaders
from model.model import get_model
from metrics import produce_submit_file


def main(args, cfg):
    eval_ids = get_data_for_evaldataset(args.eval_path_wav)

    eval_dataset = EvalDataset(eval_ids, args.eval_path_wav, pad)
    eval_dataset = {
        "eval": eval_dataset
    }
    dataloader = get_dataloaders(eval_dataset, cfg)

    model = get_model(cfg["checkpoint"], cfg["device"])
    model.load_head(cfg['knn_weights_path'])

    produce_submit_file(
        dataloader["eval"],
        model,
        cfg["device"],
        args.output_file
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--config',
                        type=str,
                        default='configs/config.json')
    parser.add_argument('--eval_path_wav',
                        type=str,
                        default="/asvspoof/wavs/")
    parser.add_argument('--output_file',
                        type=str,
                        default='submit.csv')
    args = parser.parse_args()
    config = load_checkpoint(args.config)
    main(args, config)