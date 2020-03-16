#!/usr/bin/env python
# coding: utf-8

import os
import re

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

        if (confidence >= threshold):
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


def TPR_FPR(tp, fp, p, n):
    tpr, fpr = tp / p, fp / n

    return (tpr, fpr)

pq, nq, vq, tq = questions.values()
TPR, FPR = TPR_FPR(TP, FP, valid_yes, valid_no)
TPR = round(TPR, 3)
FPR = round(FPR, 3)

print('{},{},{},{},{},{}'.format(pq, nq, vq, TPR, FPR, EER))
