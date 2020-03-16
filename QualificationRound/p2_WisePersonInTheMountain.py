#!/usr/bin/env python
# coding: utf-8

import os
import re

# Paths
root = r'/home/littlewing/Projects/MachineLearningMicrosoftPetnica/QualificationRound/p2_publicDataSet/'
inputs = r'/home/littlewing/Projects/MachineLearningMicrosoftPetnica/QualificationRound/p2_publicDataSet/inputs/'
data_set = r'/home/littlewing/Projects/MachineLearningMicrosoftPetnica/QualificationRound/p2_publicDataSet/set/'

questions = {'Yes': 0, 'No': 0, 'Valid': 0, 'Total': 0}
threshold = 0.7
TPR, FPR = 0.0, 0.0
EER = 0.0

data_set_folder = str(input())
curr_data_set = data_set + data_set_folder

regex = 'ca(\d+)\.txt|wpa(\d+)\.txt'
re_obj = re.compile(regex)

mapa_ca = dict()
mapa_wpa = dict()

mapa_valid = []

# Get ca and wpa file content
for root_, dir_, files in os.walk(curr_data_set):
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
        mapa_valid.append(k)
        questions['Valid'] += 1

for el in mapa_valid:
    pass
    # print("{:>3} : {:>3} : {:>3}".format(el, mapa_ca[el], mapa_wpa[el]))

pq, nq, vq, tq = questions.values()

print('{},{},{},{},{},{}'.format(pq, nq, vq, TPR, FPR, EER))
print(questions)
