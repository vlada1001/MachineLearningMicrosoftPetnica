import os
import re
import numpy as np

questions = {'Yes': 0, 'No': 0, 'Valid': 0, 'Total': 0}
threshold = 0.7
TP, FP, FN, TN = 0, 0, 0, 0
TPR, FPR = 0.0, 0.0
EER = 0.0

data_set_folder = input()

regex = 'ca(\d+)\.txt|wpa(\d+)\.txt'
re_obj = re.compile(regex)

mapa_ca = dict()
mapa_wpa = dict()
mapa_predictions = dict()

valid_yes = 0
valid_no = 0

'''
New BSD License

Copyright (c) 2007–2020 The scikit-learn developers.
All rights reserved.
'''

# kopirano iz sklearn.utils i modifikovano
def stable_cumsum(arr, axis=None, rtol=1e-05, atol=1e-08):
    out = np.cumsum(arr, axis=axis, dtype=np.float64)

    return out

# kopirano iz sklearn.metrics i modifikovano
def _binary_clf_curve(y_true, y_score, pos_label=None):
    y_true = np.ravel(y_true)
    y_score = np.ravel(y_score)

    if pos_label is None:
        pos_label = 1.

    y_true = (y_true == pos_label)

    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    weight = 1.

    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = stable_cumsum(y_true * weight)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    return fps, tps, y_score[threshold_idxs]


def roc_curve(y_true, y_score, pos_label=None):
    fps, tps, thresholds = _binary_clf_curve(
        y_true, y_score, pos_label=pos_label)

    if len(fps) > 2:
        optimal_idxs = np.where(np.r_[True,
                                      np.logical_or(np.diff(fps, 2),
                                                    np.diff(tps, 2)),
                                      True])[0]
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]
        thresholds = thresholds[optimal_idxs]

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[thresholds[0] + 1, thresholds]

    if fps[-1] <= 0:
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]

    return fpr, tpr, thresholds


def TPR_FPR(tp, fp, p, n):
    tpr, fpr = tp / p, fp / n

    return (tpr, fpr)


def calc_eer(mapa_ca, mapa_wpa):
    y_true = []
    y_score = []

    # If key exists in both dicts then append corresponding values
    for k in mapa_ca:
        if k in mapa_wpa:
            y_true.append(1 if mapa_ca[k] == 'Yes' else 0)
            y_score.append(mapa_wpa[k])

    fpr, tpr, threshold = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr

    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

    return round(EER, 3)


# Get ca and wpa file content
for root_, dir_, files in os.walk(data_set_folder):
    for item in files:
        # Get file path
        item_path = root_ + '/' + item

        # Enter file
        item_content = ''
        with open(item_path, 'r') as f:
            # Read file content
            item_content = f.readline().strip('% \n')

        matches = re_obj.match(item)
        if matches.group(1) == None:
            perc = float(item_content) / 100
            mapa_wpa[matches.group(2)] = perc
        else:
            mapa_ca[matches.group(1)] = str(item_content)
            questions['Total'] += 1
            questions[item_content] += 1

# Get valid questions
for k in mapa_ca:
    if k in mapa_wpa:
        questions['Valid'] += 1

        if mapa_ca[k] == 'Yes':
            valid_yes += 1
        elif mapa_ca[k] == 'No':
            valid_no += 1
        else:
            pass

        correct_answer = mapa_ca[k]
        confidence = mapa_wpa[k]
        prediction = 'False'

        if confidence >= threshold:
            prediction = 'Yes'

        if prediction == 'Yes':
            if correct_answer == 'Yes':
                TP += 1
            else:
                FP += 1
        else:
            if correct_answer == 'Yes':
                FN += 1
            else:
                TN += 1

pq, nq, vq, tq = questions.values()
TPR, FPR = TPR_FPR(TP, FP, valid_yes, valid_no)
TPR = round(TPR, 3)
FPR = round(FPR, 3)
EER = calc_eer(mapa_ca, mapa_wpa)

print('{},{},{},{},{},{}'.format(pq, nq, vq, TPR, FPR, EER))
