# -*- coding: utf-8 -*-
"""
单张图片推理脚本。

用途：给定一张汉字图片，预测其笔画序列，并还原为标准汉字。
逻辑与 train.py 中的 test() 评估流程保持一致，复用 util.py 中已有的函数，
不引入任何与训练时不同的预处理或解码方式，以保证推理结果可比。

使用前必填项：
  - WEIGHT_PATH  训练好的权重文件路径，例如 best_model.pth
  - IMAGE_PATH   待识别的单张图片路径

运行环境说明：
  - transformer.py 内部多处硬编码了 .cuda()，因此本脚本必须在具备 GPU 的环境下运行。
  - 权重由 nn.DataParallel 保存，键名带 module. 前缀，故此处同样用 DataParallel 包裹后再加载。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

from config import config
from model.transformer import Transformer
from util import (
    get_alphabet,
    tensor2str,
    rectify,
    confusing_stroke_dic,
    get_support_sample_feature_stroke,
    confusing_character_340,
)

# ============================================================
# 必填路径，使用前请填写
# ============================================================
WEIGHT_PATH = ''   # 权重文件路径，例如 './history/xxx/best_model.pth'
IMAGE_PATH = ''    # 待识别图片路径，例如 './sample/test.jpg'

# 易混字消歧所需的字体模板，来自原始 data/print_font_template 目录。
# 仅当预测出的笔画序列对应多个汉字时才会用到。若暂时不需要消歧，可保持留空。
SIMSUN_PKL = ''    # 例如 './data/print_font_template/simsun.pkl'
SIMFANG_PKL = ''   # 例如 './data/print_font_template/simfang.pkl'

# ============================================================
# 基本常量，与 util.py 中的定义对应，一般无需改动
# ============================================================
MODE = 'stroke'                          # 本项目使用笔画级识别
MAX_LENGTH = 30                          # 笔画序列最大解码步数，与 test() 中一致
mse_loss = nn.MSELoss()


def load_model(weight_path):
    """构建模型并加载权重。返回处于 eval 模式的模型。"""
    model = Transformer(MODE).cuda()
    model = nn.DataParallel(model)
    state = torch.load(weight_path)
    model.load_state_dict(state)
    model.eval()
    return model


def preprocess_image(image_path):
    """
    将单张图片转换为模型输入张量，形状为 1 x 3 x image_size x image_size。

    注意：此处的归一化方式需与训练时 data/lmdbReader.py 内 resizeNormalize 的实现保持一致。
    下方采用最常见的 ToTensor 后归一化到 [-1, 1] 的方案。
    若你的 resizeNormalize 使用了不同的均值方差，请相应修改，否则推理分布会与训练不匹配。
    """
    size = config['image_size']
    img = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),                       # 像素从 [0,255] 映射到 [0,1]
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),        # 再映射到 [-1,1]
    ])

    tensor = transform(img).unsqueeze(0)             # 增加 batch 维，得到 1 x 3 x H x W
    tensor = tensor.cuda()
    # 与 train.py 中保持一致，显式再做一次 interpolate 以确保尺寸严格正确
    tensor = F.interpolate(tensor, size=(size, size))
    return tensor


@torch.no_grad()
def predict_stroke_sequence(model, image):
    """
    自回归地逐步预测笔画序列。
    返回去除起始符与结束符后的纯笔画字符串，以及最后一层卷积特征 conv，供后续消歧使用。
    """
    alphabet = get_alphabet(MODE)
    end_index = len(alphabet) - 1            # 结束符 '$' 在字母表中的下标

    pred = torch.zeros(1, 1).long().cuda()   # 起始符，下标 0 对应 '<'
    image_features = None

    for i in range(MAX_LENGTH):
        length = torch.zeros(1).long().cuda() + i + 1
        result = model(image, length, pred, conv_feature=image_features, test=True)
        prediction = result['pred']
        now_pred = torch.max(torch.softmax(prediction, 2), 2)[1]
        pred = torch.cat((pred, now_pred[:, -1].view(-1, 1)), 1)
        image_features = result['conv']

        # 一旦预测到结束符即可停止
        if now_pred[0, -1].item() == end_index:
            break

    # pred 第 0 列是起始符，去掉；遇到结束符则截断
    raw = pred[0][1:]
    seq = []
    for idx in raw:
        if idx.item() == end_index:
            break
        seq.append(idx)

    if len(seq) == 0:
        stroke_string = ''
    else:
        stroke_string = tensor2str(MODE, torch.tensor(seq)).replace('$', '')

    return stroke_string, image_features


@torch.no_grad()
def disambiguate(model, conv_feature, stroke_string):
    """
    当一个笔画序列对应多个汉字时，用字体模板特征做最近邻匹配消歧。
    逻辑与 util.py 的 is_correct 中 stroke 分支保持一致。
    需要 SIMSUN_PKL 与 SIMFANG_PKL 两个模板文件。
    """
    if not SIMSUN_PKL or not SIMFANG_PKL:
        # 未提供模板时无法消歧，直接返回全部候选供人工判断
        return None

    confusing_dict1, gallery1 = get_support_sample_feature_stroke(model, SIMSUN_PKL)
    confusing_dict2, gallery2 = get_support_sample_feature_stroke(model, SIMFANG_PKL)

    if stroke_string not in confusing_dict1:
        return None

    probe_feature = conv_feature[0]                  # 当前图片的卷积特征
    confusing_list = confusing_dict1[stroke_string]
    gallery_feature1 = gallery1[torch.tensor(confusing_list).long()]
    gallery_feature2 = gallery2[torch.tensor(confusing_list).long()]

    scores1 = torch.tensor([mse_loss(probe_feature, g) for g in gallery_feature1]).view(-1, 1)
    scores2 = torch.tensor([mse_loss(probe_feature, g) for g in gallery_feature2]).view(-1, 1)
    scores = torch.mean(torch.cat([scores1, scores2], 1), 1)

    min_index = scores.argmin()
    return confusing_character_340[confusing_list[min_index]]


def recognize(image_path, weight_path):
    """
    完整识别流程：图片 -> 笔画序列 -> 校正 -> 还原汉字。
    返回一个包含原始笔画序列、校正后笔画序列、候选汉字与最终结果的字典。
    """
    model = load_model(weight_path)
    image = preprocess_image(image_path)

    raw_stroke, conv_feature = predict_stroke_sequence(model, image)

    # 若预测序列不在合法笔画表中，用编辑距离校正到最接近的合法序列。
    # 这一步同时承担论文中描述的 “还原非标准字” 功能。
    corrected_stroke = rectify(MODE, raw_stroke)

    result = {
        'raw_stroke_sequence': raw_stroke,
        'corrected_stroke_sequence': corrected_stroke,
        'candidates': [],
        'final_character': None,
    }

    # 校正后的笔画序列可能对应一个或多个汉字
    if corrected_stroke in confusing_stroke_dic:
        candidates = confusing_stroke_dic[corrected_stroke]
        result['candidates'] = candidates

        if len(candidates) == 1:
            result['final_character'] = candidates[0]
        else:
            # 多个候选，尝试用字体模板消歧
            disambiguated = disambiguate(model, conv_feature, corrected_stroke)
            if disambiguated is not None:
                result['final_character'] = disambiguated
            else:
                # 无法消歧时，最终结果留空，candidates 中是全部可能字
                result['final_character'] = None

    return result


if __name__ == '__main__':
    if not WEIGHT_PATH or not IMAGE_PATH:
        print('请先在脚本顶部填写 WEIGHT_PATH 与 IMAGE_PATH 后再运行。')
        raise SystemExit(0)

    output = recognize(IMAGE_PATH, WEIGHT_PATH)

    print('原始预测笔画序列 :', output['raw_stroke_sequence'])
    print('校正后笔画序列   :', output['corrected_stroke_sequence'])
    print('候选汉字         :', output['candidates'])
    print('最终识别结果     :', output['final_character'])
