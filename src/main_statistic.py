import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from scipy import stats
from scipy.ndimage import binary_dilation, binary_erosion
from sklearn.metrics import confusion_matrix, jaccard_score
from tqdm import tqdm

COLOR_DICTIONARY = {"Background": ([0, 0, 0], 0),
                    "Glosses": ([0, 255, 255], 1),
                    # "Header": ([255, 0, 255], 2),
                    "Main_Text": ([255, 255, 0], 2)
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
        class_layers[class_color_index[1]] = np.all(image_array == np.asarray(class_color_index[0]), axis=-1)

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


def rolf_metric(gt_list_unlabeled: List[int], pred_list_unlabeled: List[int]) -> float:
    conf_mat_unlabeled = confusion_matrix(gt_list_unlabeled, pred_list_unlabeled)

    # remove unlabeled class
    conf_mat_unlabeled = np.delete(conf_mat_unlabeled, 3, axis=0)
    conf_mat_unlabeled = np.delete(conf_mat_unlabeled, 3, axis=1)

    A_inter_B = conf_mat_unlabeled.diagonal()
    A = conf_mat_unlabeled.sum(1)
    B = conf_mat_unlabeled.sum(0)
    jaccard = A_inter_B / (A + B - A_inter_B)

    return jaccard.sum() / len(jaccard)


def add_unlabeled_class(img: Image) -> np.ndarray:
    img_array_ori = np.asarray(img)
    img_array = img_array_ori.copy()
    img_array[img_array != 0] = 1
    dilated = binary_dilation(img_array, iterations=1).astype(np.uint8)
    eroded = binary_erosion(img_array, iterations=1).astype(np.uint8)
    img_array = dilated - eroded
    img_array[img_array == 1] = 3
    img_array = np.where(img_array == 0, img_array_ori, img_array)
    return img_array


def evaluate_folder(method_eval_path: Path, gt_folder_path: Path):
    pred_files = sorted(list(method_eval_path.glob('*.gif')))
    gt_files = sorted(list(gt_folder_path.glob('*.gif')))
    pred_gt_list = zip(pred_files, gt_files)

    pred_list_unflatten = []
    gt_list_unflatten = []
    pred_list_unlabeled = []
    gt_list_unlabeled = []
    for pred, gt in tqdm(pred_gt_list):
        # load img
        pred_img = Image.open(pred)
        gt_img = Image.open(gt)
        if pred_img.getpalette() != gt_img.getpalette():
            print("not same palette!")
            pred_img = fix_color_table(pred_img, color_encoding=COLOR_DICTIONARY)
            gt_img = fix_color_table(gt_img, color_encoding=COLOR_DICTIONARY)

        pred_list_unlabeled.append(add_unlabeled_class(pred_img).flatten())
        gt_list_unlabeled.append(add_unlabeled_class(gt_img).flatten())

        pred_list_unflatten.append(np.asarray(pred_img).flatten())
        gt_list_unflatten.append(np.asarray(gt_img).flatten())

    pred_list_flatten = np.asarray(pred_list_unflatten).flatten()
    gt_list_flatten = np.asarray(gt_list_unflatten).flatten()
    pred_list_unlabeled = np.asarray(pred_list_unlabeled).flatten()
    gt_list_unlabeled = np.asarray(gt_list_unlabeled).flatten()

    per_page = []
    for pred, gt in zip(pred_list_unflatten, gt_list_unflatten):
        per_page.append(jaccard_score(y_pred=pred, y_true=gt, average='macro'))

    result = {'jaccard_rolf': rolf_metric(gt_list_unlabeled, pred_list_unlabeled),
              'jaccard_macro': jaccard_score(y_pred=pred_list_flatten, y_true=gt_list_flatten, average='macro'),
              'pred_list': pred_list_unflatten,
              'gt_list': gt_list_unflatten,
              'per_page': per_page
              }
    # with (pred_folder_path / 'metrics.json').open('w') as f:
    #     json.dump(result, f, indent=2)

    return result


def evaluate_folder_list(method_eval_path: Path, gt_folder_path: Path, mean_path: Path, median_path: Path,
                         method_name: str,
                         nbr_files: int):
    # get an eval file from the folder

    txt_files = [list(folder.glob('*.txt')) for folder in method_eval_path.iterdir() if folder.is_dir()]

    for exp in txt_files:
        with exp[0].open('r') as f:
            lines = f.readlines()
            results = []
            mean = []
            for line in lines:
                r = evaluate_folder(method_eval_path=Path(line.strip()) / 'test_output' / 'pred',
                                    gt_folder_path=gt_folder_path)
                mean.append(r['per_page'])
                results.append(r)
            median_network = sorted(results, key=lambda x: x['jaccard_macro'])[len(results) // 2]
            stats_line_values = median_network['per_page']
            trimmed_per_page = stats.trim_mean(mean, 0.1, axis=0)

        with median_path.open('a') as f:
            stats_str = f"{method_name}, {exp[0].parent.name}, {exp[0].parent.parent.name}," + ','.join(
                [str(x) for x in stats_line_values])
            f.write(stats_str[:-1] + '\n')
        with mean_path.open('a') as f:
            stats_str = f"{method_name}, {exp[0].parent.name}, {exp[0].parent.parent.name}," + ','.join(
                [str(x) for x in trimmed_per_page])
            f.write(stats_str[:-1] + '\n')



if __name__ == '__main__':
    # CLASSES = ['bg', 'line', 'hline', 'gline']
    CLASSES = ['bg', 'main', 'gline']

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--method_eval_path', type=Path, required=True,
                        help='e.g., /net/research-hisdoc/experiments_lars_paul/evaluations/icdar/sauvola/no_pretrain/unet')
    parser.add_argument('-o', '--output_folder_path', type=Path, required=True)
    parser.add_argument('-g', '--gt_folder_path', type=Path, required=True)
    parser.add_argument('-n', '--method_name', type=str, required=True)
    args = parser.parse_args()

    if args.method_eval_path.is_file():
        raise ValueError("pred_folder_path must be a folder!")

    statistic_file_path_median = args.output_folder_path / 'statistics' / f'{args.method_name}_median_statistics.csv'
    statistic_file_path_mean = args.output_folder_path / 'statistics' / f'{args.method_name}_mean_statistics.csv'
    file_list = sorted([i.stem for i in args.gt_folder_path.glob('*.gif')])
    nbr_files = len(file_list)
    file_str = ','.join(file_list)
    statistic_file_path_median.parent.mkdir(parents=True, exist_ok=True)
    statistic_file_path_mean.parent.mkdir(parents=True, exist_ok=True)
    if not statistic_file_path_median.exists():
        with statistic_file_path_mean.open('w') as f:
            f.write(f'method_name,epochs,training,{file_str[:-1]}\n')
        with statistic_file_path_median.open('w') as f:
            f.write(f'method_name,epochs,training,{file_str[:-1]}\n')

    evaluate_folder_list(**args.__dict__, mean_path=statistic_file_path_mean, median_path=statistic_file_path_median,
                         nbr_files=nbr_files)
