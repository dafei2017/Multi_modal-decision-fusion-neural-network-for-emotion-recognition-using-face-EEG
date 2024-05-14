# -*- coding: utf-8 -*-
"""
Created on 20221116

@author: Dafei
"""

"""
  This is an example for processing adaboost fusion algorithm in DEAP dataset using the function in ../process_adaboost/adaboost_tool
  Note that the dataset's format should be same as the format performed in 'dataset' folder
  The ONLINE dataset contains 40 trials for each subject.
  In this example, for each subject, 20 trials are for training whereas the other trials are for testing. 
"""

import sys
sys.path.append('../process_adaboost')
import numpy as np
import adaboost_tool
import os
os.environ['CUDA_VISIBLE_DEVICES'] = 'gup:0'

if __name__ == '__main__':
    ROOT_PATH = '../../dataset/DEAP2/'
    
    for subject_id in range(1, 15):
        subject_path = ROOT_PATH + str(subject_id)+'/'
        
        adaboost_model = adaboost_tool.Adaboost_model(preprocessed=True)
        #random select 20 trial for training, the other trials for testing
        train_idxs_set = set(np.random.choice(np.arange(1, 41), size = 20, replace = False))
        all_set = set(np.arange(1, 41))
        test_idxs_set = all_set - train_idxs_set
        
        #training
        for trial_id in train_idxs_set:
            
            trial_path = subject_path + 'trial_' + str(trial_id) + '/'
            # if os.path.isfile(trial_path + 'faces.npy'):
            #     adaboost_model.add_one_trial_data(trial_path)
            # else:
            #     pass
            adaboost_model.add_one_trial_data(trial_path)


        adaboost_model.train()
        
        #testing
        acc_valence, acc_arousal = 0., 0.
        for trial_id in test_idxs_set:
            trial_path = subject_path + 'trial_' + str(trial_id) + '/'
            valence_correct, arousal_correct = adaboost_model.predict_one_trial(trial_path)
            acc_valence += 1 if valence_correct else 0
            acc_arousal += 1 if arousal_correct else 0
        
        print (subject_id, acc_valence/20, acc_arousal/20)
