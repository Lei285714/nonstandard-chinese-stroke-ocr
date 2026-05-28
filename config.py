# -*- coding: utf-8 -*-
# 训练与测试的全局配置。
# 下方凡是标注 “路径，需按本地环境修改” 的项，使用前都必须改成你自己机器上的真实路径。
# 其余参数除非你清楚其作用，否则建议保持默认。

config = {
    # 实验名称。每次运行会在 ./history/ 下创建同名文件夹存放本次的权重与日志。
    # 注意 util.py 的 saver() 会在运行开始时删除同名旧文件夹，重名会覆盖此前结果，请谨慎复用。
    'exp_name': 'I_love_OCR',

    # 训练总轮数。仅训练时生效，test_only 为 True 时不起作用。
    'epoch': 114514,

    # 学习率。配合 Adadelta 优化器使用，Adadelta 的初始 lr 通常设为 1.0。
    'lr': 1.0,

    # 识别粒度。'stroke' 为本项目使用的笔画级方案，'character' 为字符级对照方案。
    'mode': 'stroke',  # 可选 character 或 stroke

    # 每个批次的样本数。显存不足时调小。
    'batch': 24,

    # 训练过程中每隔多少个 iteration 触发一次验证。
    'val_frequency': 1000,

    # 是否只做测试不做训练。True 表示加载 resume 指定的权重直接在测试集上评估。
    'test_only': True,

    # 预训练权重路径，路径，需按本地环境修改。
    # 留空字符串则从零开始训练；填写 best_model.pth 路径则加载已训练权重。
    # 该权重由 nn.DataParallel 保存，键名带 module. 前缀，加载时需保持一致。
    'resume': '',

    # 训练集 LMDB 路径，路径，需按本地环境修改。
    # 支持用英文逗号分隔填写多个数据集。LMDB 由 jpg_to_lmdb.py 生成。
    'train_dataset': '',

    # 测试集 LMDB 路径，路径，需按本地环境修改。同样支持逗号分隔的多个路径。
    'test_dataset': '',

    # 是否启用权重衰减。True 时优化器附加 1e-4 的 weight_decay。
    'weight_decay': False,

    # 每隔多少 epoch 将学习率乘以 0.1。当前设得极大，等同于不做学习率衰减。
    'schedule_frequency': 1000000,

    # 输入图像边长。本项目统一缩放为 64x64，与论文中受显存限制的下采样设置一致。
    # 若修改此值，需同步确认 LMDB 内图像尺寸与之匹配。
    'image_size': 64,

    # 字表规模。3755 对应 GB2312 一级常用汉字数量，与 util.py 中的字表长度对应。
    'alphabet': 3755,
}
