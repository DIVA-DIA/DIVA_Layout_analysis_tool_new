from dataclasses import dataclass, field
from pathlib import Path
import csv
from typing import List

from scipy import stats
import numpy as np


@dataclass
class Method:
    name: str
    epochs: int
    training: str
    page_values: List[float] = field(default_factory=list)


if __name__ == '__main__':
    baseline_path = Path(
        "/net/research-hisdoc/experiments_lars_paul/evaluations/icdar/statistics/baseline_mean_statistics.csv")
    exp_path = Path(
        "/net/research-hisdoc/experiments_lars_paul/evaluations/icdar/statistics/sauvola_mean_statistics.csv")
    t_test_path = Path(
        "/net/research-hisdoc/experiments_lars_paul/evaluations/icdar/statistics/t-tests/sauvola_mean_statistics.csv")

    trainings = ['training-20', 'training-10', 'training-5']
    pre_train_epochs = ['10', '20', '30', '40', '50']

    with t_test_path.open('w') as f:
        f.write('training,epochs,t,p\n')

    baselines = []
    with baseline_path.open('r') as f:
        reader = csv.reader(f)
        for line in reader:
            if line[0] == 'method_name' or int(line[1]) != 100:
                continue
            baselines.append(Method(line[0], int(line[1]), line[2], np.asarray(line[3:]).astype(np.float64)))

    exp = []
    with exp_path.open('r') as f:
        reader = csv.reader(f)
        for line in reader:
            if line[0] == 'method_name' or line[1].strip() not in pre_train_epochs or line[2].strip() not in trainings:
                continue
            exp.append(Method(line[0], int(line[1]), line[2], np.asarray(line[3:]).astype(np.float64)))

    for baseline in baselines:
        experiments = [e for e in exp if e.training == baseline.training]
        for e in experiments:
            t_test = stats.ttest_ind(baseline.page_values, e.page_values)
            with t_test_path.open('a') as f:
                f.write(f'{e.training},{e.epochs},{t_test[0]},{t_test[1]}\n')
