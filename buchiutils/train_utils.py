import os
import csv

def save_train_hyper_csv(hyper_dict, filename):
    assert isinstance(filename, str)
    assert isinstance(hyper_dict, dict)
    with open(filename, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=list(hyper_dict.keys()))
        writer.writeheader()
        writer.writerow(hyper_dict)

def save_train_progress(progress_dict, filename, reset = False):
    assert isinstance(filename, str)
    assert isinstance(progress_dict, dict)
    if reset:
        with open(filename, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=list(progress_dict.keys()))
            writer.writeheader()
            writer.writerow(progress_dict)
    else:
        with open(filename, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=list(progress_dict.keys()))
            writer.writerow(progress_dict)
            