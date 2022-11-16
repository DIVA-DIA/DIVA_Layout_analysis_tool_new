import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm
from PIL import Image
import numpy as np

COLOR_DICTIONARY = {"Background": ([0, 0, 0], 0),
                    "Glosses": ([0, 255, 255], 1),
                    "Header": ([255, 0, 255], 2),
                    "Main_Text": ([255, 255, 0], 3)
                    }


def _create_pillow_color_table(color_dict: Dict[str, Tuple[List[int], int]]) -> List[int]:
    color_table = [0] * 768
    for name, color_and_index in color_dict.items():
        color, index = color_and_index
        color_table[index * 3] = color[0]
        color_table[index * 3 + 1] = color[1]
        color_table[index * 3 + 2] = color[2]
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


def main(folder_path: Path, output_path: Path, color_encoding: Dict[str, Tuple[List[int], int]]):
    if color_encoding is None:
        color_encoding = COLOR_DICTIONARY

    output_path.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(list(folder_path.glob('*.gif'))):
        img = Image.open(img_path)
        fixed_img = fix_color_table(img, color_encoding)
        fixed_img.save(output_path / img_path.name)

    with open(output_path / 'class_encoding.json', 'w') as f:
        json.dump(color_encoding, fp=f)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--folder_path', type=Path, required=True,
                        help='Path to the folder to images to be fixed')
    parser.add_argument('-o', '--output_path', type=Path, required=True, help='Path to the output folder')
    parser.add_argument('-c', '--color_encoding', type=Path, required=False, default=None,
                        help='Path to the color encoding json file')
    args = parser.parse_args()

    main(**args.__dict__)
