# In[119]:


#!/usr/bin/env python
# coding: utf-8

'''
    Treba da rekurzivno prodjem kroz sve foldere i ucitam putanje wpa#.txt i ca#.txt
'''

import numpy as np
import os
import re
from sklearn import metrics

# In[120]:

# Paths
root = r'/home/littlewing/Projects/MachineLearningMicrosoftPetnica/QualificationRound/p2_publicDataSet/'
inputs = r'/home/littlewing/Projects/MachineLearningMicrosoftPetnica/QualificationRound/p2_publicDataSet/inputs/'
outputs = r'/home/littlewing/Projects/MachineLearningMicrosoftPetnica/QualificationRound/p2_publicDataSet/outputs/'
data_set = r'/home/littlewing/Projects/MachineLearningMicrosoftPetnica/QualificationRound/p2_publicDataSet/set/'


# In[121]:


threshold = 0.7

# data_set_folder = input()
# curr_data_set = data_set + data_set_folder

regex = 'ca(\d+)\.txt|wpa(\d+)\.txt'
re_obj = re.compile(regex)


# In[169]:


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
    
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    
    return round(EER, 3)



# In[170]:


def print_output(res_list):
    print('{:>5}  {:>5}  {:>5}  {:>5}  {:>5}  {:>5}'.format(
        'pq', 'nq', 'vq', 'TPR', 'FPR', 'EER'))

    for res in res_list:
        pq, nq, vq, TPR, FPR, EER, file = res
        print('{:>5}  {:>5}  {:>5}  {:>5}  {:>5}  {:>5} : {:>6}'.format(
            pq, nq, vq, TPR, FPR, EER, file))


def output_diff(output, expected_output):
    output_diff = []

    for i in range(len(output)):
        row = []

        for el in range(len(output[0]) - 1):
            abs_diff = abs(float(output[i][el]) -
                           float(expected_output[i][el]))
            row.append(round(abs_diff, 3))

        row.append(output[i][-1])
        output_diff.append(row)
    return output_diff


# In[171]:


def output(input_root):
    output = []

    for input_file in os.listdir(input_root):
        curr_data_set = data_set + input_file.split('.txt')[0]
        file = input_file

        questions = {'Yes': 0, 'No': 0, 'Valid': 0, 'Total': 0}
        TP, FP, FN, TN = 0, 0, 0, 0
        TPR, FPR = 0.0, 0.0
        EER = 0.0

        mapa_ca = dict()
        mapa_wpa = dict()

        mapa_valid = []

        valid_yes = 0
        valid_no = 0

        # get ca and wpa file content
        for root_, dir_, files_ in os.walk(curr_data_set):
            for item in files_:
                # get file path
                item_path = root_ + '/' + item

                # enter file
                item_content = ''
                with open(item_path, 'r') as f:
                    # read file content
                    item_content = f.readline().strip('% \n')

                matches = re_obj.match(item)
                if matches.group(1) == None:
                    perc = float(item_content) / 100
                    mapa_wpa[matches.group(2)] = perc
                else:
                    mapa_ca[matches.group(1)] = str(item_content)
                    questions['Total'] += 1
                    questions[item_content] += 1

        # Get valid questions and calculate TP, FP, FN, TN
        for k in mapa_ca:
            if k in mapa_wpa:
                mapa_valid.append(k)
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

        pq, nq, vq, tq = questions.values()

        TPR, FPR = TPR_FPR(TP, FP, valid_yes, valid_no)
        TPR = round(TPR, 3)
        FPR = round(FPR, 3)
        EER = calc_eer(mapa_ca, mapa_wpa)

        output.append([pq, nq, vq, TPR, FPR, EER, file])

    return output


# In[172]:


def expected_output(output_root):
    expected_output = []

    for output_file in os.listdir(output_root):
        curr_output_file = output_root + output_file

        with open(curr_output_file, 'r') as f:
            pq = nq = vq = TPR = FPR = EER = 0

            item_content = f.readline().strip(' ,\n')
            row = item_content.split(',')
            pq = int(row[0])
            nq = int(row[1])
            vq = int(row[2])
            TPR = float(row[3])
            FPR = float(row[4])
            EER = float(row[5])

            row = [pq, nq, vq, TPR, FPR, EER, output_file]
            expected_output.append(row)

    return expected_output


# In[173]:


print('output:')
output = output(inputs)
print_output(output)

print('\nexpected output:')
expected_output = expected_output(outputs)
print_output(expected_output)

print('\ndiff:')
output_diff = output_diff(output, expected_output)
print_output(output_diff)


# In[ ]:


# In[ ]:
