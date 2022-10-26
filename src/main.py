import json
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sn
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

COLOR_DICTIONARY = {"Background": ([0, 0, 0], 0),
                    "Glosses": ([0, 255, 255], 1),
                    "Header": ([255, 0, 255], 2),
                    "Main_Text": ([255, 255, 0], 3)
                    }


def _get_class_layers(image_array: np.ndarray, color_map: Dict[str, List[int]]) -> Dict[str, np.ndarray]:
    class_layers = {}
    for class_name, class_color in color_map.items():
        class_layers[class_name] = np.asarray([[all(cell == class_color) for cell in row] for row in image_array])

    return class_layers


def _create_pillow_color_table(color_dict: Dict[str, Tuple[List[int], int]]) -> List[int]:
    color_table = []
    for name, color_and_index in color_dict.items():
        color, index = color_and_index
        color_table.extend(color)
    return color_table


def _create_class_layers(image_array: np.ndarray, color_map: Dict[str, List[int]]) -> Dict[int, np.ndarray]:
    class_layers = {}
    for class_name, class_color_index in color_map.items():
        class_layers[class_color_index[1]] = np.asarray(
            [[all(cell == class_color_index[0]) for cell in row] for row in image_array])

    return class_layers


def fix_color_table(img, color_encoding: Dict[str, Tuple[List[int], int]]):
    img_array_2d = np.asarray(img)
    img_array_3d = np.asarray(img.convert(mode='RGB'))
    new_img_array = np.empty(shape=img_array_2d.shape, dtype=np.uint8)
    # set max of np.unit8(255) to have error check
    new_img_array[:] = 255
    class_masks = _create_class_layers(img_array_3d, color_encoding)
    for class_index, class_mask in class_masks.items():
        new_img_array[class_mask] = class_index
    new_img = Image.fromarray(new_img_array, mode='P')
    new_img.putpalette(_create_pillow_color_table(color_encoding))
    return new_img


def main(pred_folder_path: Path, gt_folder_path: Path):
    pred_files = sorted(list(pred_folder_path.glob('*.gif')))
    gt_files = sorted(list(gt_folder_path.glob('*.gif')))
    pred_gt_list = zip(pred_files, gt_files)

    i = 0
    pred_list = []
    gt_list = []
    for pred, gt in tqdm(pred_gt_list):
        # load img
        pred_img = Image.open(pred)
        gt_img = Image.open(gt)
        if pred_img.getpalette() != gt_img.getpalette():
            print("not same palette!")
            pred_img = fix_color_table(pred_img, color_encoding=COLOR_DICTIONARY)
            gt_img = fix_color_table(gt_img, color_encoding=COLOR_DICTIONARY)

        if i == 0:
            pred_list = np.asarray(pred_img)
            gt_list = np.asarray(gt_img)
        else:
            pred_list = np.concatenate((np.asarray(pred_img), pred_list))
            gt_list = np.concatenate((np.asarray(gt_img), gt_list))

        i += 1

    pred_list = pred_list.flatten()
    gt_list = gt_list.flatten()

    conf_mat = confusion_matrix(gt_list, pred_list)
    conf_mat_norm = confusion_matrix(gt_list, pred_list, normalize='true')

    plt.figure(figsize=(14, 8))
    # set labels size
    sn.set(font_scale=1.4)
    # set font size
    fig = sn.heatmap(conf_mat_norm, annot=True, annot_kws={"size": 8}, fmt="g")
    for i in range(conf_mat_norm.shape[0]):
        fig.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor='yellow', lw=3))
    plt.ylabel('Predictions')
    plt.xlabel('Targets')
    plt.title("test")
    plot_folder_path = pred_folder_path / "plots"
    plot_folder_path.mkdir(exist_ok=True)
    plt.savefig(plot_folder_path / 'conf_mat.png')

    # Per-class accuracy
    class_accuracy = 100 * conf_mat.diagonal() / conf_mat.sum(1)

    A_inter_B = conf_mat.diagonal()
    A = conf_mat.sum(1)
    B = conf_mat.sum(0)
    jaccard = A_inter_B / (A + B - A_inter_B)

    result = {'conf_mat': conf_mat.tolist(),
              'class_accuracy': class_accuracy.tolist(),
              'jaccard': jaccard.tolist(),
              'jaccard_mean': jaccard.mean().tolist(),
              'classification_report': classification_report(gt_list, pred_list, target_names=CLASSES, output_dict=True),
              }
    with (pred_folder_path / 'metrics.json').open('w') as f:
        json.dump(result, f, indent=2)


if __name__ == '__main__':
    CLASSES = ['bg', 'line', 'hline', 'gline']

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred_folder_path', type=Path, required=True)
    parser.add_argument('-g', '--gt_folder_path', type=Path, required=True)
    args = parser.parse_args()

    main(**args.__dict__)
    pass
