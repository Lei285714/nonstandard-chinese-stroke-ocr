# Recognition and Restoration of Chinese Non-Standard Characters Based on Stroke Decomposition

[中文版 README](./README.md)

This project is a tool for recognizing and restoring **non-standard, irregular handwritten Chinese characters**. The core idea is to decompose each character into a standardized stroke sequence: the model first predicts the stroke sequence, then restores it to a standard character through lexicon matching. This makes recognition possible even when the glyph is distorted, strokes are connected, or the spatial structure is disordered.

---

## Attribution and Provenance

This project is not entirely original. It is a derivative and re-training effort built upon the work of the Fudan University team, which is stated prominently here:

- The method and base code come from the `stroke-level-decomposition` module of **FudanVI/FudanOCR**, corresponding to the paper *Zero-Shot Chinese Character Recognition with Stroke-Level Decomposition* by Chen et al., IJCAI 2021. Original repository: https://github.com/FudanVI/FudanOCR/tree/main/stroke-level-decomposition
- The contribution of this project is to retrain the model using the **Irregular Handwritten Chinese Character Dataset, IHCCD** by Ji et al., 2024, migrating and validating a method originally designed for zero-shot recognition of multi-font printed characters on the task of recognizing and restoring non-standard handwritten characters.
- The `document/` directory retains the PDFs of the original papers. The base code under `data/` and `model/` is inherited from the original repository.

If you use this project in your research, please cite both references listed in the Citation section below.

---

## Method Overview

The recognition pipeline consists of three sequential modules:

1. **Image-to-Feature Encoder**: Built on a ResNet backbone, it converts the input character image into a feature map and extracts basic information such as the direction and shape of strokes.
2. **Feature-to-Stroke Decoder**: Using a Transformer decoder structure, it translates the feature map into a stroke sequence. Strokes are grouped into five basic types: horizontal, vertical, left-falling, right-falling, and turning.
3. **Stroke-to-Character Decoder**: It retrieves the character corresponding to the stroke sequence from a lexicon. If there is a direct match, that character is output. If not, the edit distance is used to find the closest valid stroke sequence, a step that also serves the **restoration of non-standard characters**. For the one-to-many case where a single stroke sequence maps to multiple characters, nearest-neighbor matching on font-template features is used for disambiguation.

By decomposing characters down to strokes, the smallest constituent unit, the model does not depend on whether a complete character or a particular radical was seen during training. This gives it the ability to generalize to unseen characters and unseen radicals, and naturally accommodates the diversity of non-standard glyphs.

---

## Experimental Results

| Setting | Training Data | Test Data | CACC |
| --- | --- | --- | --- |
| Direct transfer, no non-standard training | Printed Artistic | IHCCD | 25.22% |
| Baseline ResNet | HWDB1.1 | IHCCD | 24.23% |
| Retrained on non-standard characters | IHCCD | IHCCD | **77.16%** |

One point to state honestly: due to limited GPU memory, the third experiment downsampled all images in IHCCD from the original 128×128 to 64×64 before training and testing. This compression very likely affected the final accuracy. The 77.16% figure was obtained under this constrained setting and does not represent the upper bound of the model's capability. Retraining at higher resolution with more GPU memory is expected to improve it further.

---

## Model Weights Download

The weight file `best_model.pth` is provided with the repository via Git LFS. If pulling large files from GitHub is inconvenient, you can also obtain it from the following drives:

- OneDrive：[best_model.pth](https://1drv.ms/u/c/574ef7ae453129f3/IQDH_UOPbWVZRqxasHorERhKAXVRTdKKTZTGXTi2ep8rN5k?e=zgzAqK)
- 夸克网盘：https://pan.quark.cn/s/fa443f3ec04a

After downloading, place it back into the `history/0713TestOnIHCCD/` directory, or point `WEIGHT_PATH` in `inference.py` to wherever you store it.

---

## Project Structure

```
stroke-level-decomposition/
├── config.py              Global configuration; paths must be adapted to your environment
├── inference.py           Single-image inference script
├── jpg_to_lmdb.py         Data preprocessing; converts a JPG dataset to LMDB
├── train.py               Main training and testing program
├── util.py                Data loading, encoding/decoding, disambiguation utilities
├── requirement.txt        Dependency list
│
├── data/
│   ├── decompose-stroke-3755.txt          Stroke decomposition for 3755 common characters, required
│   ├── decompose-stroke-27533.txt         Stroke decomposition for a larger character set
│   ├── decompose-radical-27533.txt        Radical decomposition, for comparison
│   ├── decompose-stroke-korean-2350.txt   Korean decomposition, for comparison
│   ├── lmdbReader.py                       LMDB data reading and normalization
│   ├── chinese_character_test.lmdb/        Sample test data
│   └── print_font_template/
│       ├── simsun.pkl                      SimSun template, for confusable-character disambiguation
│       └── simfang.pkl                     SimFang template, for confusable-character disambiguation
│
├── document/              Original paper PDFs
├── history/               Training artifacts: the weights best_model.pth plus a backup of the original code as it was during training. Note that the versions outside this folder have minor readability edits that do not affect execution
│   └── 0713TestOnIHCCD/
│
└── model/
    └── transformer.py     Model architecture; ResNet encoder and Transformer decoder
```

---

## Requirements

```
Python 3.8
PyTorch 1.10.0
CUDA 11.1
cuDNN 8.0.5
```

Remaining dependencies are listed in `requirement.txt`. Install them with:

```bash
pip install -r requirement.txt
```

Pay particular attention to the `python-Levenshtein` dependency, used for the edit-distance computation on stroke sequences and easily overlooked.

**On the running device**: the current code hardcodes `.cuda()` in several places, so it requires an NVIDIA GPU by default. If your machine has no NVIDIA GPU, you will first need to rewrite the device-related code in `model/transformer.py`, `util.py`, and `inference.py` into a device-adaptive form that can fall back to CPU. The model itself is small, and single-image CPU inference takes only a few seconds on an ordinary computer.

---

## Usage

### 1. Data Preparation

`jpg_to_lmdb.py` converts a JPG dataset organized as "class folder / images" into the LMDB format the model requires, and resizes all images to a uniform size. The dataset directory should be structured as:

```
dataset_root/
├── charA/
│   ├── img1.jpg
│   └── img2.jpg
├── charB/
│   └── ...
```

Edit `jpg_dir`, `lmdb_path`, and `img_size` at the bottom of the script, then run:

```bash
python jpg_to_lmdb.py
```

Note that `img_size` must match `image_size` in `config.py`, which is 64 in this project.

### 2. Training

In `config.py`, set `test_only` to `False`, fill in the `train_dataset` and `test_dataset` paths, leave `resume` empty to train from scratch, then run:

```bash
python train.py
```

Training artifacts are saved under `./history/experiment_name/`.

### 3. Testing and Evaluation

In `config.py`, set `test_only` to `True`, point `resume` at a trained weight file, fill in `test_dataset`, then run:

```bash
python train.py
```

The console prints per-sample results in real time in the following format:

```
sample_index | corrected_prediction | ground_truth | is_correct | prediction_probability | running_accuracy | raw_prediction
```

### 4. Single-Image Inference

`inference.py` recognizes a single image. Open the script and fill in the two required paths at the top:

```python
WEIGHT_PATH = ''   # Weight path, e.g. ./history/0713TestOnIHCCD/best_model.pth
IMAGE_PATH = ''    # Path to the image to recognize
```

To disambiguate confusable characters, also fill in the two font-template paths. If left empty, all candidate characters are listed when a one-to-many case is encountered:

```python
SIMSUN_PKL = ''    # ./data/print_font_template/simsun.pkl
SIMFANG_PKL = ''   # ./data/print_font_template/simfang.pkl
```

Then run:

```bash
python inference.py
```

The output includes the raw stroke sequence, the corrected stroke sequence, the candidate characters, and the final recognition result.

---

## Known Issues and Notes

- **Device dependency**: As noted above, the original code requires an NVIDIA GPU by default. Running locally without a GPU requires a device-adaptive rewrite first.
- **`must_in_screen()` in `util.py`**: `train.py` requires execution inside a Linux screen session, otherwise it exits immediately. This was added by the original author to prevent training interruptions. On a local Windows machine, or when this restriction is unwanted, comment out the call to this function in `train.py`. `inference.py` does not impose this restriction.
- **`saver()` in `util.py`**: At the start of training, it deletes any existing folder under `./history/` that shares the current `exp_name`. Reusing an experiment name will **silently overwrite** previous weights and logs, so name carefully.
- **Preprocessing consistency**: The image normalization in `inference.py` must exactly match the implementation of `resizeNormalize` in `data/lmdbReader.py`. Otherwise the inference-time data distribution will not match training, significantly lowering accuracy. Verify that file before making changes.
- **PyTorch version**: The code uses the legacy `dataloader.next()` style, which is deprecated in newer PyTorch versions and must be changed to `next(dataloader)`. Setting up the environment with the versions specified above is recommended.

---

## Citation

If this project helps your research, please cite the original method paper and the dataset used:

```bibtex
@inproceedings{chen2021zero,
  title={Zero-Shot Chinese Character Recognition with Stroke-Level Decomposition},
  author={Jingye Chen and Bin Li and Xiangyang Xue},
  booktitle={IJCAI},
  year={2021}
}

@article{ji2024ihccd,
  title={IHCCD: dataset for identification of irregular handwritten Chinese characters},
  author={Ji, Jiamei and Shao, Yunxue and Ji, Tanzheng},
  journal={Journal of Image and Graphics},
  volume={29},
  number={11},
  pages={3345--3356},
  year={2024},
  doi={10.11834/jig.230047}
}
```

---

## Acknowledgments

Sincere thanks to the FudanVI team at Fudan University for open-sourcing the stroke-decomposition method and code, and to the team at Nanjing Tech University for building and releasing the IHCCD dataset. This project is a migration and re-training effort built on top of their work.

---

## License

The upstream repository FudanVI/FudanOCR does not declare an explicit open-source license. When using, modifying, or redistributing this project, please respect the rights of the upstream work and prioritize academic research purposes. For other uses, it is advisable to confirm licensing with the upstream authors first.
