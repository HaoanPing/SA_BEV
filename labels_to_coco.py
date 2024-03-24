import os
import cv2
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', default='./',type=str, help="root path of images and labels, include ./images and ./labels and classes.txt")
parser.add_argument('--save_path', type=str,default='annotations/instance_train2017', help="if not split the dataset, give a path to a json file")

arg = parser.parse_args()

def read_split_files(split_dir):
    split_files = ['train.txt', 'val.txt', 'test.txt']
    split_data = {}
    for file_name in split_files:
        with open(os.path.join(split_dir, file_name), 'r') as file:
            images = [line.strip() for line in file.readlines()]
            split_data[file_name.split('.')[0]] = images
    return split_data

def yolo2coco(arg):
    root_path = arg.root_dir
    print("Loading data from ",root_path)

    assert os.path.exists(root_path)
    originLabelsDir = os.path.join(root_path, 'labels')
    originImagesDir = os.path.join(root_path, 'images')
    with open(os.path.join(root_path, 'classes.txt')) as f:
        classes = f.read().strip().split()

    # 读取分割数据
    split_data = read_split_files(root_path)

    train_img, val_img, test_img = split_data['train'], split_data['val'], split_data['test']

    datasets = {
        'train': {'categories': [], 'annotations': [], 'images': []},
        'val': {'categories': [], 'annotations': [], 'images': []},
        'test': {'categories': [], 'annotations': [], 'images': []}
    }

    for i, cls in enumerate(classes, 1):  # 从1开始编号以符合COCO格式
        category = {'id': i, 'name': cls, 'supercategory': 'mark'}
        for dataset in datasets.values():
            dataset['categories'].append(category)
    # 标注的id
    ann_id_cnt = 0
    for dataset_name, image_filenames in [('train', train_img), ('val', val_img), ('test', test_img)]:
        for img_filename in tqdm(image_filenames, desc=f"Processing {dataset_name}"):
            img_path = os.path.join(originImagesDir, img_filename + '.jpg')
            im = cv2.imread(img_path)
            height, width, _ = im.shape
            image_id = len(datasets[dataset_name]['images']) + 1
            datasets[dataset_name]['images'].append({
                'file_name': img_filename + '.jpg',
                'id': image_id,
                'width': width,
                'height': height
            })

            label_file = img_filename + '.txt'
            label_path = os.path.join(originLabelsDir, label_file)
            if os.path.exists(label_path):
                with open(label_path, 'r') as fr:
                    for line in fr.readlines():
                        cls_id, x_center, y_center, bbox_width, bbox_height = [float(x) for x in line.strip().split()]
                        x1 = (x_center - bbox_width / 2) * width
                        y1 = (y_center - bbox_height / 2) * height
                        x2 = (x_center + bbox_width / 2) * width
                        y2 = (y_center + bbox_height / 2) * height
                        datasets[dataset_name]['annotations'].append({
                            'id': ann_id_cnt,
                            'image_id': image_id,
                            'category_id': int(cls_id),
                            'bbox': [x1, y1, x2 - x1, y2 - y1],
                            'area': (x2 - x1) * (y2 - y1),
                            'iscrowd': 0,
                            'segmentation': []
                        })
                        ann_id_cnt += 1

        # 保存结果
        for dataset_name in ['train', 'val', 'test']:
            with open(os.path.join(root_path, f'annotations_{dataset_name}.json'), 'w') as f:
                json.dump(datasets[dataset_name], f, indent=4)

        print('COCO format conversion completed.')

if __name__ == "__main__":
    yolo2coco(arg)
