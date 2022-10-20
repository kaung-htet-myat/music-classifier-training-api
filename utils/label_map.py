import json


def get_label_map(label_path):
    with open(label_path, 'r', encoding='utf-8') as label_file:
        labels = json.load(label_file)
    reverse_labels = {v:k for k, v in labels.items()}

    return labels, reverse_labels