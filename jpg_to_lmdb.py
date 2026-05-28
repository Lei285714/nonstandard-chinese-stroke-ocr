import os
import lmdb
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from io import BytesIO  # 导入BytesIO
import sys

def jpg_to_lmdb(jpg_dir, lmdb_path, img_size=None):
    """
    将JPG图像数据集转换为LMDB格式
    
    参数:
    jpg_dir: JPG图像所在目录，结构应为: 根目录/类别/图像.jpg
    lmdb_path: 输出LMDB文件的路径
    img_size: 可选，图像统一缩放尺寸，例如(128, 128)
    """
    # 创建LMDB环境，调整map_size为2GB
    env = lmdb.open(lmdb_path, map_size=2147483648)  # 2GB
    txn = env.begin(write=True)
    image_count = 1
    
    # 遍历所有图像文件
    for root, _, files in os.walk(jpg_dir):
        safe_root = root.encode('utf-8', errors='replace').decode('utf-8')
        # 使用replace替换乱码字符为下划线，确保显示正常
        display_root = safe_root.replace('�', '_')
        for file in tqdm(files, desc=f"Processing {safe_root}"):
            if file.lower().endswith(('.jpg', '.jpeg')):
                try:
                    # 读取图像
                    img_path = os.path.join(root, file)
                    img = Image.open(img_path)
                    
                    # 可选：调整图像大小
                    if img_size:
                        img = img.resize(img_size, Image.BILINEAR)
                    
                    # 修复：使用BytesIO处理图像字节流
                    img_byte_arr = BytesIO()
                    img.save(img_byte_arr, format='JPEG')
                    img_bytes = img_byte_arr.getvalue()
                    
                    # 获取标签（文件夹名）
                    label = os.path.basename(root)
                    label_bytes = label.encode('utf-8', errors='replace')
                    
                    # 构建键值对 (key: image_00001, value: 图像字节)
                    # (key: label_00001, value: 标签)
                    key_img = f"image_{image_count:09d}".encode()
                    key_label = f"label_{image_count:09d}".encode()
                    
                    txn.put(key_img, img_bytes)
                    txn.put(key_label, label_bytes)
                    image_count += 1
                    
                    # 每1000张提交一次
                    if image_count % 1000 == 0:
                        txn.commit()
                        txn = env.begin(write=True)
                except Exception as e:
                    # Handle paths with invalid Unicode characters
                    safe_path = img_path.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                    print(f"Error processing {safe_path}: {e}")
    
    # 保存总图像数
    txn.put(b'num-samples', str(image_count - 1).encode())
    txn.commit()
    env.close()
    print(f"成功转换 {image_count} 张图像到 LMDB 数据库")

# 使用示例
if __name__ == "__main__":
    jpg_to_lmdb(
        jpg_dir="",  # JPG数据集路径
        lmdb_path="",     # 输出LMDB路径
        img_size=(64, 64)                  # 可选：统一图像尺寸
    )