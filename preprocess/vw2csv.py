# -*- coding:utf-8 -*-
"""
vowpal wabbit to csv
"""

import pandas as pd
import codecs
import csv


def to_dict(line):
    """
    file line to dict
    :param line:
    :return:
    """
    result = dict()
    fields = line.split('|')
    # uid
    # result['uid'] = str(fields[0].split()[1])
    # age
    # result['age'] = int(fields[1].split()[1])
    # gender
    # result['gender'] = int(fields[2].split()[1])
    # marriageStatus
    # result['marriageStatus'] = fields[3].split()[1:]

    for field in fields:
        result[field.split()[0]] = ' '.join(field.split()[1:])
    return result


head = ['uid', 'age', 'gender', 'marriageStatus', 'education', 'consumptionAbility', 'LBS', 'interest1', 'interest2',
        'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3', 'appIdInstall',
        'appIdAction',
        'ct', 'os', 'carrier', 'house']
file = codecs.open('../input/userFeature.data', encoding='utf-8')

with open('../input/userFeature.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=head)
    writer.writeheader()
    for i, line in enumerate(file):
        # results.append(to_dict(line))
        writer.writerow(to_dict(line))
        if i % 10000 == 0:
            print('converted line ' + str(i))
