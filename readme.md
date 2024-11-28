# SafeSpeak-2024 Hackathon approach

This repository provides code for solution __SafeSpeak-2024 competition__. The main idea is to use wav2vec 2.0 as feature extractor, AASIST + KNN-head as detector. 
<!-- 
### Submit

To make submit file for competition we provide __submit.py__:
```bash
python submit.py --config configs/config_res2tcnguard.json --eval_path_wav wavs_dir
``` -->

## Installation
First, clone the repository locally, create and activate a conda environment, and install the requirements :
```
$ git clone https://github.com/Georgyshul/SafeSpeak_approach.git
$ conda create -n spoof python=3.7
$ conda activate spoof
$ pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
$ cd fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1
(This fairseq folder can also be downloaded from https://github.com/pytorch/fairseq/tree/a54021305d6b3c4c5959ac9395135f63202db8f1)
$ pip install --editable ./
$ pip install -r requirements.txt
```


## Data preparation
The validation and evaluation are done on ASVspoof19 LA dataset [1].

ASVspoof2019 dataset: https://datashare.ed.ac.uk/handle/10283/3336
  1. Download `LA.zip` and unzip it
  2. Set your dataset and labels directories and files in the corresponding variables in `configs/config.json` file: `train_path_flac`(`dev_path_flac`,`eval_path_flac`) and `train_label_path`(`dev_label_path`,`eval_label_path`)

## Training KNN 
To make predictions and train new KNN-head you should use [eval.py](https://github.com/Georgyshul/SafeSpeak_approach/blob/main/eval.py):
```bash
cd SafeSpeak_approach

python eval.py --config configs/config.json --train_knn True
```

## Submit
To evaluate the model on the test data you should use [submit.py](https://github.com/Georgyshul/SafeSpeak_approach/blob/main/submit.py):
```bash
cd SafeSpeak_approach

python submit.py --config configs/config.json --eval_path_wav <path_to_your_data>
```


## Pre-trained model
We provide models checkpoints:


| Model           | Weights                                                                                                   |
|-----------------|-----------------------------------------------------------------------------------------------------------|
| **XLS-R**       | [XLS-R-300M](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec/xlsr)
| **AASIST**      | [LA_model.pth](https://drive.google.com/drive/folders/1c4ywztEVlYVijfwbGLl9OEa1SNtFKppB)
| **KNN**         | [knn_*.bin](https://drive.google.com/drive/folders/1NjH8SXdyom1A1n63oT_JVij-PyS4nZho?usp=drive_link) 

Don't forget to specify paths to weights in `config.json` file.

### License
```
MIT License

Copyright (c) 2024 MTUCI 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Acknowledgements
The dataset we use is ASVspoof 2019 [1]
- https://www.asvspoof.org/index2019.html

### References
[1] ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech
```bibtex
@article{wang2020asvspoof,
  title={ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech},
  author={Wang, Xin and Yamagishi, Junichi and Todisco, Massimiliano and Delgado, H{\'e}ctor and Nautsch, Andreas and Evans, Nicholas and Sahidullah, Md and Vestman, Ville and Kinnunen, Tomi and Lee, Kong Aik and others},
  journal={Computer Speech \& Language},
  volume={64},
  pages={101114},
  year={2020},
  publisher={Elsevier}
}
```
[2] AASIST backbone
```bibtex

@inproceedings{tak2022automatic,
  title={Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and data augmentation},
  author={Tak, Hemlata and Todisco, Massimiliano and Wang, Xin and Jung, Jee-weon and Yamagishi, Junichi and Evans, Nicholas},
  booktitle={The Speaker and Language Recognition Workshop},
  year={2022}
}
```
[3] [Code baseline](https://github.com/mtuciru/SafeSpeak-2024)

[4] XLS-R
```bibtex
@article{babu2021xlsr,
      title={XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale}, 
      author={Arun Babu and Changhan Wang and Andros Tjandra and Kushal Lakhotia and Qiantong Xu and Naman Goyal and Kritika Singh and Patrick von Platen and Yatharth Saraf and Juan Pino and Alexei Baevski and Alexis Conneau and Michael Auli},
      year={2021},
      volume={abs/2111.09296},
      journal={arXiv},
}
```
