import json
import os
import cv2

# 指定 JSON 文件路径
json_file_path = 'annotations/sample_2D_annotations.json'

source_dir = 'D:images'

# 指定输出目录
output_dir = 'D:labels'
os.makedirs(output_dir, exist_ok=True)  # 创建输出目录（如果不存在）

# 读取 JSON 文件
with open(json_file_path, 'r') as file:
    data = json.load(file)

# 准备一个字典来收集同一张图片中的所有目标框信息
annotations_by_image = {}

for item in data:
    filename = os.path.basename(item['filename'])
    bbox = item['bbox_corners']
    img_path = os.path.join(source_dir, filename)  # 图片的路径
    img = cv2.imread(img_path)
    img_height, img_width = img.shape[:2]

    # 计算中心坐标和宽高（归一化）
    x_center = (bbox[0] + bbox[2]) / 2 / img_width
    y_center = (bbox[1] + bbox[3]) / 2 / img_height
    bbox_width = (bbox[2] - bbox[0]) / img_width
    bbox_height = (bbox[3] - bbox[1]) / img_height

    # 构建目标框信息字符串
    bbox_str = f'0 {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}'

    # 将目标框信息添加到对应图片的列表中
    if filename not in annotations_by_image:
        annotations_by_image[filename] = []
    annotations_by_image[filename].append(bbox_str)

# 遍历收集到的每张图片的目标框信息，写入对应的.txt文件
for filename, bbox_strs in annotations_by_image.items():
    filename_without_ext, _ = os.path.splitext(filename)
    output_path = os.path.join(output_dir, f'{filename_without_ext}.txt')
    with open(output_path, 'w') as f:
        for bbox_str in bbox_strs:
            f.write(f'{bbox_str}\n')

print("转换完成。")