# 基于笔画分解的中文非标准字符识别与还原
# Recognition and Restoration of Chinese Non-Standard Characters Based on Stroke Decomposition

[English README](./README-EN.md)

本项目是一个面向**非标准、不规范手写汉字**的识别与还原工具。核心思路是把每个汉字分解为标准化的笔画序列，先由模型预测笔画序列，再通过词典匹配把序列还原为标准汉字，从而在字形扭曲、笔画粘连、结构错乱等情况下仍能完成识别。

---

## 衍生关系与来源声明

本项目并非完全原创，而是建立在复旦大学团队工作之上的衍生与再训练成果，特此显著声明：

- 方法与基础代码来自 **FudanVI/FudanOCR** 的 `stroke-level-decomposition` 模块，对应论文 *Zero-Shot Chinese Character Recognition with Stroke-Level Decomposition*（Chen et al., IJCAI 2021）。原始仓库地址：https://github.com/FudanVI/FudanOCR/tree/main/stroke-level-decomposition
- 本项目的贡献在于：使用**非规范手写汉字数据集 IHCCD**（Ji et al., 2024）对模型重新训练，将原本面向多字体印刷体零样本识别的方法，迁移并验证到非标准手写汉字的识别与还原任务上。
- 仓库内 `document/` 目录保留了原论文的 PDF，`data/` 与 `model/` 下的基础代码沿用自原仓库。

如果你在研究中使用了本项目，请同时引用下方"引用"一节中的两篇文献。

---

## 方法概述

整个识别流程由三个串联的模块构成：

1. **图像到特征编码器（Image-to-Feature Encoder）**：以 ResNet 为骨干，将输入的汉字图像转换为特征图，提取笔画的方向、形状等基础信息。
2. **特征到笔画解码器（Feature-to-Stroke Decoder）**：采用 Transformer 解码器结构，将特征图翻译为笔画序列。笔画被归纳为五种基本类型，分别为横、竖、撇、捺、折。
3. **笔画到字符解码器（Stroke-to-Character Decoder）**：在词典中检索笔画序列对应的汉字。若直接命中则输出该字；若未命中，则用编辑距离找到最接近的合法笔画序列，这一步同时承担**还原非标准字**的功能。对于同一笔画序列对应多个汉字的"一对多"情况，再借助字体模板特征做最近邻匹配以消歧。

把汉字拆到笔画这一最小单位，使模型不依赖于训练集中是否见过某个完整字或某个部首，因此具备对未见字、未见部首的泛化能力，也天然适配非标准字形的多样性。

---

## 实验结果

| 设置 | 训练数据 | 测试数据 | CACC |
| --- | --- | --- | --- |
| 直接迁移，未用非标准字训练 | Printed Artistic | IHCCD | 25.22% |
| 对照基线 ResNet | HWDB1.1 | IHCCD | 24.23% |
| 用非标准字重新训练 | IHCCD | IHCCD | **77.16%** |

需要诚实说明的一点：受显存限制，第三组实验把 IHCCD 中所有图像从原始的 128×128 下采样到了 64×64 之后再训练与测试。这一压缩很可能影响了最终精度，77.16% 是在该受限设置下取得的结果，并非模型能力的上限。后续在更大显存、更高分辨率下重新训练有望进一步提升。

---

## 模型权重下载

权重文件 `best_model.pth` 已随仓库通过 Git LFS 一同提供。若你不便从 GitHub 拉取大文件，也可通过以下网盘获取：

- OneDrive：[best_model.pth](https://1drv.ms/u/c/574ef7ae453129f3/IQDH_UOPbWVZRqxasHorERhKAXVRTdKKTZTGXTi2ep8rN5k?e=zgzAqK)
- 夸克网盘：https://pan.quark.cn/s/fa443f3ec04a

下载后将其放回 `history/0713TestOnIHCCD/` 目录，或在 `inference.py` 的 `WEIGHT_PATH` 中指向你的实际存放路径即可。

---

## 项目结构

```
stroke-level-decomposition/
├── config.py              全局配置，路径等需按本地环境修改
├── inference.py           单张图片推理脚本
├── jpg_to_lmdb.py         数据预处理，将 JPG 数据集转为 LMDB
├── train.py              训练与测试主程序
├── util.py               数据加载、编解码、消歧等工具函数
├── requirement.txt        依赖列表
│
├── data/
│   ├── decompose-stroke-3755.txt          3755 常用字的笔画分解表，运行必需
│   ├── decompose-stroke-27533.txt         更大字表的笔画分解
│   ├── decompose-radical-27533.txt        部首分解，对照用
│   ├── decompose-stroke-korean-2350.txt   韩文分解，对照用
│   ├── lmdbReader.py                       LMDB 数据读取与归一化
│   ├── chinese_character_test.lmdb/        示例测试数据
│   └── print_font_template/
│       ├── simsun.pkl                      宋体模板，易混字消歧用
│       └── simfang.pkl                     仿宋模板，易混字消歧用
│
├── document/              原论文 PDF
├── history/               训练产物，含权重 best_model.pth及训练时的原始代码备份，请注意现在在这个文件夹外的版本做过一些小的可读性上的修改，不影响运行
│   └── 0713TestOnIHCCD/
│
└── model/
    └── transformer.py     模型结构，ResNet 编码器与 Transformer 解码器
```

---

## 环境要求

```
Python 3.8
PyTorch 1.10.0
CUDA 11.1
cuDNN 8.0.5
```

其余依赖见 `requirement.txt`，用以下命令安装：

```bash
pip install -r requirement.txt
```

请特别留意 `python-Levenshtein` 这一依赖，它用于笔画序列的编辑距离计算，容易被遗漏。

**关于运行设备**：当前代码在多处硬编码了 `.cuda()`，因此默认必须在具备 NVIDIA GPU 的环境下运行。若你的机器没有 NVIDIA 显卡，需要先将 `model/transformer.py`、`util.py`、`inference.py` 中的设备相关代码改写为自适应形式，使其能够回退到 CPU。模型本身不大，单张图片的 CPU 推理在普通电脑上也只需数秒。

---

## 使用方法

### 一、数据准备

`jpg_to_lmdb.py` 把按"类别文件夹 / 图片"组织的 JPG 数据集转换为模型所需的 LMDB 格式，并统一缩放尺寸。数据集目录结构应为：

```
数据集根目录/
├── 字A/
│   ├── 图片1.jpg
│   └── 图片2.jpg
├── 字B/
│   └── ...
```

修改脚本底部的 `jpg_dir`、`lmdb_path`、`img_size` 后运行：

```bash
python jpg_to_lmdb.py
```

注意 `img_size` 需与 `config.py` 中的 `image_size` 保持一致，本项目为 64。

### 二、训练

在 `config.py` 中把 `test_only` 设为 `False`，填好 `train_dataset` 与 `test_dataset` 路径，将 `resume` 留空表示从零训练，然后运行：

```bash
python train.py
```

训练产物会保存在 `./history/实验名/` 下。

### 三、测试与评估

在 `config.py` 中把 `test_only` 设为 `True`，将 `resume` 指向已训练好的权重，填好 `test_dataset`，然后运行：

```bash
python train.py
```

控制台会按以下格式实时输出每个样本的结果：

```
样本序号 | 校正后预测 | 正确标签 | 是否正确 | 预测概率 | 累计准确率 | 原始预测
```

### 四、单张图片推理

`inference.py` 用于对单张图片做识别。打开脚本，填写顶部的两项必填路径：

```python
WEIGHT_PATH = ''   # 权重路径，例如 ./history/0713TestOnIHCCD/best_model.pth
IMAGE_PATH = ''    # 待识别图片路径
```

若需要对易混字消歧，再填写两个字体模板路径，不填则会在遇到一对多时列出全部候选字：

```python
SIMSUN_PKL = ''    # ./data/print_font_template/simsun.pkl
SIMFANG_PKL = ''   # ./data/print_font_template/simfang.pkl
```

然后运行：

```bash
python inference.py
```

输出包含原始笔画序列、校正后笔画序列、候选汉字与最终识别结果。

---

## 已知问题与注意事项

- **设备依赖**：如上所述，原始代码默认需要 NVIDIA GPU，本地无显卡需先做设备自适应改造。
- **`util.py` 的 `must_in_screen()`**：`train.py` 要求必须在 Linux 的 screen 会话中运行，否则会直接退出。这是原作者为防止训练中断而设。在 Windows 本地或不希望此限制时，可注释掉 `train.py` 中对该函数的调用。`inference.py` 未引入此限制。
- **`util.py` 的 `saver()`**：训练开始时会删除 `./history/` 下与当前 `exp_name` 同名的旧文件夹。复用实验名会**静默覆盖**此前的权重与日志，请谨慎命名。
- **预处理一致性**：`inference.py` 的图像归一化方式必须与 `data/lmdbReader.py` 中 `resizeNormalize` 的实现完全一致，否则推理时的数据分布与训练不匹配，会显著拉低准确率。修改前请核对该文件。
- **PyTorch 版本**：代码中使用了 `dataloader.next()` 这一旧式写法，在较新版本的 PyTorch 中已废弃，需改为 `next(dataloader)`。建议按上方指定版本搭建环境。

---

## 引用

如果本项目对你的研究有帮助，请引用原始方法论文与所用数据集：

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

## 致谢

衷心感谢复旦大学 FudanVI 团队开源的笔画分解方法与代码，感谢南京工业大学团队构建并公开 IHCCD 数据集。本项目是在这些工作的基础上完成的迁移与再训练尝试。

---

## 许可证

原始上游仓库 FudanVI/FudanOCR 未声明明确的开源许可证。在使用、修改或再分发本项目时，请一并尊重上游工作的权利，并优先用于学术研究目的。如需将本项目用于其他用途，建议先与上游作者确认授权情况。
