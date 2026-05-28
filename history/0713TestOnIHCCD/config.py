config = {
    'exp_name': '0713TestOnIHCCD_lhy',
    'epoch': 300,
    'lr': 1.0,
    'mode': 'stroke',  # character / stroke
    'batch': 24,
    'val_frequency': 1000,
    'test_only': True,
    'resume': '/root/FudanOCR/stroke-level-decomposition/history/0704TrainOnIHCCD_lhy/best_model.pth',
    'train_dataset': '/root/autodl-tmp/mydata/IHCCDtrainlmdb',
    'test_dataset': '/root/autodl-tmp/mydata/IHCCDtestlmdb3',
    'weight_decay': False,
    'schedule_frequency': 1000000,
    'image_size': 64,
    'alphabet': 3755,
}
