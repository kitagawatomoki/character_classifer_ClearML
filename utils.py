import os
import random

# def get_etl_train_val_data(char_list):
#     train_list = []
#     val_list = []
#     for char in char_list:
#         path = "../etl_paths/train/{}.txt".format(char)
#         if os.path.exists(path):
#             with open(path) as f:
#                 char_paths = ["{} {}".format(char, s.strip()) for s in f.readlines()]
#             train_list+=char_paths

#         path = "../etl_paths/val/{}.txt".format(char)
#         if os.path.exists(path):
#             with open(path) as f:
#                 char_paths = ["{} {}".format(char, s.strip()) for s in f.readlines()]
#             val_list+=char_paths

#     return train_list, val_list

# def get_etl_test_data(char_list):
#     test_list = []
#     for char in char_list:
#         path = "../etl_paths/test/{}.txt".format(char)
#         if os.path.exists(path):
#             with open(path) as f:
#                 char_paths = ["{} {}".format(char, s.strip()) for s in f.readlines()]
#             test_list+=char_paths

#     return test_list

# def get_gen_data(use_gen, char_list, num_use=None):
#     gen_list = []
#     for char in char_list:
#         path = "{}/{}.txt".format(use_gen, char)
#         if os.path.exists(path):
#             with open(path) as f:
#                 char_paths = ["{} {}".format(char, s.strip()) for s in f.readlines()]
#             if num_use is not None:
#                 char_paths = random.sample(char_paths, num_use)
#             gen_list+=char_paths
#     return gen_list


from multiprocessing import Pool, Manager
#
# ETLの訓練，検証データの取得
#
def load_etl_train_val_data(char):
    train_paths, val_paths = [], []

    train_path = f"../etl_paths/train/{char}.txt"
    if os.path.exists(train_path):
        with open(train_path) as f:
            train_paths.extend(f"{char} {line.strip()}" for line in f)

    val_path = f"../etl_paths/val/{char}.txt"
    if os.path.exists(val_path):
        with open(val_path) as f:
            val_paths.extend(f"{char} {line.strip()}" for line in f)

    return train_paths, val_paths

def get_etl_train_val_data(char_list):
    num_processes = 16
    with Pool(processes=num_processes) as pool:
        results = pool.map(load_etl_train_val_data, char_list)

    # 結果を結合する
    train_list = sum((train for train, _ in results), [])
    val_list = sum((val for _, val in results), [])

    return train_list, val_list

#
# ETLの評価データの取得
#
def load_etl_test_data(char):
    test_paths = []

    test_path = f"../etl_paths/test/{char}.txt"
    if os.path.exists(test_path):
        with open(test_path) as f:
            test_paths.extend(f"{char} {line.strip()}" for line in f)

    return test_paths

def get_etl_test_data(char_list):
    num_processes = 16
    with Pool(processes=num_processes) as pool:
        results = pool.map(load_etl_test_data, char_list)

    # 結果を結合する
    test_list = sum((test for test in results), [])

    return test_list

#
# 生成画像データの取得
#
def load_gen_data(info):
    use_gen, char, num_use = info[0], info[1], info[2]
    gen_paths = []

    test_path = f"{use_gen}/{char}.txt"
    if os.path.exists(test_path):
        with open(test_path) as f:
            gen_paths.extend(f"{char} {line.strip()}" for line in f)
        if num_use is not None:
            gen_paths = random.sample(gen_paths, num_use)

    return gen_paths

def get_gen_data(use_gen, char_list, num_use=None):
    num_processes = 16
    info = [[use_gen, char, num_use] for char in char_list]
    with Pool(processes=num_processes) as pool:
        results = pool.map(load_gen_data, info)

    # 結果を結合する
    test_list = sum((test for test in results), [])

    return test_list


from sklearn.metrics import classification_report,accuracy_score
import numpy as np

def calculate_precision_recall(array1, array2, target_value):
    # True Positive (TP): 両方の配列でtarget_valueと一致する要素
    TP = np.sum((array1 == target_value) & (array2 == target_value))

    # False Positive (FP): array1ではtarget_valueでないが、array2ではtarget_valueである要素
    FP = np.sum((array1 != target_value) & (array2 == target_value))

    # False Negative (FN): array1ではtarget_valueで、array2ではtarget_valueでない要素
    FN = np.sum((array1 == target_value) & (array2 != target_value))

    # Precision と Recall を計算
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    return precision, recall

def get_classification_report(all_char, chars, label_list, pred_list):
    w = "　        | precision | recall | f1-score | support"
    report = [w]
    f1_list = []
    for idx, c in enumerate(chars):
        label_idx = all_char.index(c)

        precision, recall = calculate_precision_recall(np.array(label_list), np.array(pred_list), label_idx)
        f1 = (2*precision*recall)/(precision+recall)

        w = "       {} |    {:.4f} | {:.4f} |   {:.4f} |     {}".format(c, precision, recall, f1, label_list.count(label_idx))
        report.append(w)
        f1_list.append(f1)

    macro_F1 = sum(f1_list)/len(f1_list)
    accuracy = accuracy_score(label_list, pred_list)
    w = ""
    report.append(w)
    w = "accuracy                           {:.4f} |  {}".format(accuracy, len(label_list))
    report.append(w)
    w = "macro f1                           {:.4f} |  {}".format(macro_F1, len(label_list))
    report.append(w)

    return report
