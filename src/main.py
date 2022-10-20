from typing import Dict, List

from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor
from torchmetrics import F1Score, JaccardIndex, Precision, Recall


def _get_class_layers(image_array: np.ndarray, color_map: Dict[str, List[int]]) -> Dict[str, np.ndarray]:
    class_layers = {}
    for class_name, class_color in color_map.items():
        class_layers[class_name] = np.asarray([[all(cell == class_color) for cell in row] for row in image_array])

    return class_layers


if __name__ == '__main__':
    color_map_o = {'background': [0, 0, 0],
                   'text_line': [255, 255, 0],
                   'header': [255, 0, 255],
                   'comment': [0, 255, 255]}
    # load image and gt
    prediction = Image.open('data/pred/fmb-cb-55-011r.gif')
    gt = Image.open('data/pred/fmb-cb-55-011r.gif')
    prediction = prediction.convert('RGB')
    gt = gt.convert('RGB')

    pred_tensor = ToTensor()(prediction)
    gt_tensor = ToTensor()(gt)

    # create conf mat
    # get jaccard index
    # get F1 score
    f1score = F1Score(num_classes=4)
    # get precision
    # get recall
    pass
