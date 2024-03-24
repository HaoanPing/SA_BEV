import random
import argparse
import json
import os
from collections import OrderedDict
from typing import List, Tuple, Union
import numpy as np
from pyquaternion.quaternion import Quaternion
from shapely.geometry import MultiPoint, box
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
import cv2


def post_process_coords(corner_coords: List,
                        imsize: Tuple[int, int] = (1600, 900)) -> Union[Tuple[float, float, float, float], None]:
    """
    Get the intersection of the convex hull of the reprojected bbox corners and the image canvas, return None if no
    intersection.
    :param corner_coords: Corner coordinates of reprojected bounding box.
    :param imsize: Size of the image canvas.
    :return: Intersection of the convex hull of the 2D box corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def generate_record(ann_rec: dict,
                    x1: float,
                    y1: float,
                    x2: float,
                    y2: float,
                    sample_data_token: str,
                    filename: str) -> OrderedDict:
    """
    Generate one 2D annotation record given various informations on top of the 2D bounding box coordinates.
    :param ann_rec: Original 3d annotation record.
    :param x1: Minimum value of the x coordinate.
    :param y1: Minimum value of the y coordinate.
    :param x2: Maximum value of the x coordinate.
    :param y2: Maximum value of the y coordinate.
    :param sample_data_token: Sample data token.
    :param filename:The corresponding image file where the annotation is present.
    :return: A sample 2D annotation record.
    """
    repro_rec = OrderedDict()
    repro_rec['sample_data_token'] = sample_data_token

    relevant_keys = [
        'attribute_tokens',
        'category_name',
        'instance_token',
        'next',
        'num_lidar_pts',
        'num_radar_pts',
        'prev',
        'sample_annotation_token',
        'sample_data_token',
        'visibility_token',
    ]

    for key, value in ann_rec.items():
        if key in relevant_keys:
            repro_rec[key] = value

    repro_rec['bbox_corners'] = [x1, y1, x2, y2]
    repro_rec['filename'] = filename

    return repro_rec


def get_2d_boxes(sample_data_token: str, visibilities: List[str]) -> List[OrderedDict]:
    """
    Get the 2D annotation records for a given `sample_data_token`.
    :param sample_data_token: Sample data token belonging to a camera keyframe.
    :param visibilities: Visibility filter.
    :return: List of 2D annotation record that belongs to the input `sample_data_token`
    """

    # Get the sample data and the sample corresponding to that sample data.
    sd_rec = nusc.get('sample_data', sample_data_token)

    assert sd_rec['sensor_modality'] == 'camera', 'Error: get_2d_boxes only works for camera sample_data!'
    if not sd_rec['is_key_frame']:
        raise ValueError('The 2D re-projections are available only for keyframes.')

    s_rec = nusc.get('sample', sd_rec['sample_token'])

    # Get the calibrated sensor and ego pose record to get the transformation matrices.
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

    # Get all the annotation with the specified visibilties.
    ann_recs = [nusc.get('sample_annotation', token) for token in s_rec['anns']]
    ann_recs = [ann_rec for ann_rec in ann_recs if (ann_rec['visibility_token'] in visibilities)]

    repro_recs = []

    for ann_rec in ann_recs:
        # Augment sample_annotation with token information.
        ann_rec['sample_annotation_token'] = ann_rec['token']
        ann_rec['sample_data_token'] = sample_data_token

        # Get the box in global coordinates.
        box = nusc.get_box(ann_rec['token'])

        # Move them to the ego-pose frame.
        box.translate(-np.array(pose_rec['translation']))
        box.rotate(Quaternion(pose_rec['rotation']).inverse)

        # Move them to the calibrated sensor frame.
        box.translate(-np.array(cs_rec['translation']))
        box.rotate(Quaternion(cs_rec['rotation']).inverse)

        # Filter out the corners that are not in front of the calibrated sensor.
        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        corner_coords = view_points(corners_3d, camera_intrinsic, True).T[:, :2].tolist()

        # Keep only corners that fall within the image.
        final_coords = post_process_coords(corner_coords)

        # Skip if the convex hull of the re-projected corners does not intersect the image canvas.
        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords

        # Generate dictionary record to be included in the .json file.
        repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y, sample_data_token, sd_rec['filename'])
        repro_recs.append(repro_rec)

    return repro_recs


def make_classes(nuscenes, output_file_path):
    with open(output_file_path, 'w') as file:
        for idx, category in enumerate(nuscenes.category):
            class_str = f"{category['name']}\n"
            file.write(class_str)
    print(f"类别信息已保存到 {output_file_path}")


def generate_split_lists(source_dir, split_ratios=(0.8, 0.1, 0.1), split_names=('train', 'val', 'test'), filter_cond='CAM'):
    images = []
    source_dir = os.path.join(source_dir, 'samples')
    for subdir, _, files in os.walk(source_dir):
        if filter_cond in os.path.basename(subdir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    images.append(os.path.join(subdir, file))

    random.shuffle(images)
    total_images = len(images)
    split_points = [int(ratio * total_images) for ratio in split_ratios]
    split_points = [0] + [sum(split_points[:i+1]) for i in range(len(split_points))]

    split_datasets = [images[split_points[i]:split_points[i+1]] for i in range(len(split_ratios))]

    for names, dataset in zip(split_names, split_datasets):
        with open(f"{names}.txt", 'w') as f:
            for img_name in dataset:
                f.write(f"{img_name}\n")
    print("训练、验证、测试集划分文件已生成。")


def make_labels(json_file_path, img_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # 创建输出目录（如果不存在）

    # 读取 JSON 文件
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    annotations_by_image = {}  # 准备一个字典来收集同一张图片中的所有目标框信息

    for item in tqdm(data, desc=f"Processing {json_file_path}"):
        filename = os.path.basename(item['filename'])
        bbox = item['bbox_corners']
        img_path = os.path.join(img_dir, item['filename'])
        img = cv2.imread(img_path)
        if img is None:
            continue  # 如果图片读取失败，则跳过
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


def read_split_files(split_dir):
    split_files = ['train.txt', 'val.txt', 'test.txt']
    split_data = {}
    for file_name in split_files:
        with open(os.path.join(split_dir, file_name), 'r') as file:
            images = [line.strip() for line in file.readlines()]
            split_data[file_name.split('.')[0]] = images
    return split_data



def yolo2coco(root_path):
    assert os.path.exists(root_path)
    originLabelsDir = 'labels'
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
    for dataset_name, image_paths in [('train', train_img), ('val', val_img), ('test', test_img)]:
        for img_path in tqdm(image_paths, desc=f"Processing {dataset_name}"):
            im = cv2.imread(img_path)
            height, width, _ = im.shape
            image_id = len(datasets[dataset_name]['images']) + 1
            datasets[dataset_name]['images'].append({
                'file_name': img_path,
                'id': image_id,
                'width': width,
                'height': height
            })

            label_file = os.path.splitext(img_path)[0] + '.txt'
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
            with open(os.path.join(root_path, 'annotations', f'instances_{dataset_name}2017.json'), 'w') as f:
                json.dump(datasets[dataset_name], f, indent=4)

    print('COCO format conversion completed.')

def main(args):

    print("Generating 2D reprojections of the nuScenes dataset")

    # Get tokens for all camera images.
    sample_data_camera_tokens = [s['token'] for s in nusc.sample_data if (s['sensor_modality'] == 'camera') and
                                 s['is_key_frame']]

    # For debugging purposes: Only produce the first n images.
    if args.image_limit != -1:
        sample_data_camera_tokens = sample_data_camera_tokens[:args.image_limit]

    # Loop through the records and apply the re-projection algorithm.
    reprojections = []
    for token in tqdm(sample_data_camera_tokens):
        reprojection_records = get_2d_boxes(token, args.visibilities)
        reprojections.extend(reprojection_records)

    # Save to a .json file.
    dest_path = os.path.join(args.dataroot, args.version)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    with open(args.json_path, 'w') as fh:
        json.dump(reprojections, fh, sort_keys=True, indent=4)
    print("Saved the 2D re-projections under {}".format(args.json_path))
    make_classes(nusc, args.classes_file)
    generate_split_lists(args.dataroot)
    make_labels(args.json_path, args.dataroot, 'labels')
    yolo2coco('./')
    print("NuScenes data processing complete.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simplified NuScenes dataset processing.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--version', type=str, default='../nuscenes_mini/v1.0-mini',
                        help='Version of the NuScenes dataset.')
    parser.add_argument('--dataroot', type=str, default='../nuscenes_mini',
                        help='Root directory of the NuScenes dataset.')
    parser.add_argument('--json_path', type=str, default='annotations/sample_2D_annotations.json',
                        help='Output json_filename.')
    parser.add_argument('--visibilities', type=str, default=['', '1', '2', '3', '4'],
                        help='Visibility bins, the higher the number the higher the visibility.', nargs='+')
    parser.add_argument('--image_limit', type=int, default=-1,
                        help='Number of images to process or -1 to process all.')
    parser.add_argument('--classes_file', type=str, default='classes.txt',
                        help='Output file for classes.')
    parser.add_argument('--images_dir', type=str, default='images',
                        help='Directory to store filtered images.')
    parser.add_argument('--images_with_annotations_dir', type=str, default='images_with_annotations',
                        help='Where to save annotated images.')

    args = parser.parse_args()
    nusc = NuScenes(dataroot=args.dataroot, version=args.version)
    # 这里你需要根据你自己的代码逻辑修改 process_dataset 函数的参数，以及确保它执行了你想要的操作
    main(args)