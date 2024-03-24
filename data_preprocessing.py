import shutil
import random
from PIL import Image, ImageDraw
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


def copy_images_with_filter(source_dir, target_dir, filter_cond='CAM'):
    os.makedirs(target_dir, exist_ok=True)
    for subdir, _, files in os.walk(source_dir):
        if filter_cond in os.path.basename(subdir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    shutil.copy(os.path.join(subdir, file), os.path.join(target_dir, file))
    print("图片复制完成。")


def generate_split_lists(target_dir, split_ratios=(0.8, 0.1, 0.1), split_names=('train', 'val', 'test')):
    images = [os.path.splitext(file)[0] for file in os.listdir(target_dir) if file.lower().endswith('.jpg')]
    random.shuffle(images)

    split_points = [int(ratio * len(images)) for ratio in split_ratios]
    split_datasets = [images[sum(split_points[:i]):sum(split_points[:i+1])] for i in range(len(split_points))]

    for names, dataset in zip(split_names, split_datasets):
        with open(f"{names}.txt", 'w') as f:
            for img_name in dataset:
                f.write(f"{img_name}\n")
    print("训练、验证、测试集划分文件已生成。")


def move_images_based_on_split(datadir, split_dirs, split_files):
    for split_dir, split_file in zip(split_dirs, split_files):
        os.makedirs(split_dir, exist_ok=True)
        with open(split_file, 'r') as f:
            for line in f:
                file_name = line.strip() + '.jpg'
                shutil.copy(os.path.join(datadir, file_name), split_dir)


def draw_annotations_from_json_and_make_labels(json_file_path, source_dir, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('labels', exist_ok=True)

    # 读取 JSON 文件
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # 根据 filename 对检测框数据进行分组
    detections_by_image = {}
    for item in data:
        filename = item['filename']
        detections_by_image.setdefault(filename, []).append(item)

    # 使用tqdm显示进度
    for filename in tqdm(detections_by_image.keys(), desc="Processing images"):
        detections = detections_by_image[filename]
        img_path = os.path.join(source_dir, os.path.basename(filename))  # 图片的路径
        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)

        # 绘制当前图片的所有检测框
        labels_set = []
        for detection in detections:
            bbox = detection['bbox_corners']
            draw.rectangle(bbox, outline="red", width=2)
            img_width, img_height = img.size
            x_center = (bbox[0] + bbox[2]) / 2 / img_width
            y_center = (bbox[1] + bbox[3]) / 2 / img_height
            bbox_width = (bbox[2] - bbox[0]) / img_width
            bbox_height = (bbox[3] - bbox[1]) / img_height
            bbox_str = f'0 {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}'
            labels_set.append(bbox_str)

        # 保存处理后的图片到指定的输出目录
        save_path = os.path.join(output_dir, os.path.basename(filename))
        img.save(save_path)
        filename_without_ext, _ = os.path.splitext(os.path.basename(filename))
        with open(os.path.join('labels', f'{filename_without_ext}.txt'), 'w') as f:
            for bbox_str in labels_set:
                f.write(f'{bbox_str}\n')

    print(f"Annotation drawing completed. Annotated images saved in '{output_dir}'.")


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
    copy_images_with_filter(os.path.join(args.dataroot, 'samples'), args.images_dir)
    generate_split_lists(args.images_dir)
    move_images_based_on_split(args.images_dir, ['train2017', 'val2017', 'test2017'], ['train.txt', 'val.txt', 'test.txt'])
    # draw_annotations_from_json_and_make_labels(args.json_path, args.images_dir, args.images_with_annotations_dir)
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