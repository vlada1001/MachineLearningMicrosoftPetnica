#!/usr/bin/env python
# coding: utf-8

import os, fnmatch, re

# Paths
root = r'/home/littlewing/Projects/MachineLearningMicrosoftPetnica/QualificationRound/p2_publicDataSet/'
inputs = r'/home/littlewing/Projects/MachineLearningMicrosoftPetnica/QualificationRound/p2_publicDataSet/inputs/'
data_set = r'/home/littlewing/Projects/MachineLearningMicrosoftPetnica/QualificationRound/p2_publicDataSet/set/'

questions = {'Yes': 0, 'No': 0, 'Valid': 0, 'Total': 0}
threshold = 0.7
TPR, FPR = 0, 0
EER = 0

curr_data_set = data_set + '1'

regex = 'ca(\d+)\.txt|wpa(\d+)\.txt'
re_obj = re.compile(regex)

mapa_ca = dict()
mapa_wap = dict()

# get ca and wap file content
for root_, dir_, files in os.walk(curr_data_set):
    for item in files:
        # get file path
        item_path = root_ + '/' + item

        # enter file
        item_content = ''
        with open(item_path, 'r') as f:
            # read file content
            item_content = f.readline().strip(' ').strip('\n').strip('%')

        matches = re_obj.match(item)
        if matches.group(1) == None:
            mapa_wap[matches.group(2)] = float(item_content) / 100
        else:
            mapa_ca[matches.group(1)] = str(item_content)
            questions['Total'] += 1
            questions[item_content] += 1

print(questions)
                