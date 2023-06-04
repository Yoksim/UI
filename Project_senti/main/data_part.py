# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 11:02:23 2021

@author: zhendahu
"""

import pandas as pd
import numpy as np

data = pd.read_excel('C:/Users/zhendahu/Desktop/Data/0test-ready-check-SH.xlsx')

#data_part = data[(data['Correct2_A'] == 'yes') & (data['Correct2_B'] == 'yes') & (data['Correct2_C'] == 'yes') & (data['Correct2_D'] == 'yes')]


data_part = data[((data['Correct5_A'] == 'yes') & (data['Correct5_B'] == 'yes') & (data['Correct5_C'] == 'yes'))
                 | ((data['Correct5_B'] == 'yes') & (data['Correct5_C'] == 'yes') & (data['Correct5_D'] == 'yes'))
                 | ((data['Correct5_A'] == 'yes') & (data['Correct5_C'] == 'yes') & (data['Correct5_D'] == 'yes'))
                 | ((data['Correct5_A'] == 'yes') & (data['Correct5_B'] == 'yes') & (data['Correct5_D'] == 'yes'))]

data_part.to_csv('C:/Users/zhendahu/Desktop/Data/data_part.csv')