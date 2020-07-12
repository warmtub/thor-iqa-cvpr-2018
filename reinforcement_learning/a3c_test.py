# -*- coding: utf-8 -*-
import pdb
import tensorflow as tf
import os
import threading
import numpy as np
import h5py
import glob

import random

from networks.qa_planner_network import QAPlannerNetwork
from networks.free_space_network import FreeSpaceNetwork
from networks.end_to_end_baseline_network import EndToEndBaselineNetwork
from reinforcement_learning.a3c_testing_thread import A3CTestingThread 
from utils import tf_util
from utils import py_util

import constants
np.set_printoptions(threshold=np.inf)

def main():
    if constants.OBJECT_DETECTION:
        from darknet_object_detection import detector
        detector.setup_detectors(constants.PARALLEL_SIZE)

    with tf.device('/gpu:' + str(constants.GPU_ID)):
        with tf.variable_scope('global_network'):
            if constants.END_TO_END_BASELINE:
                global_network = EndToEndBaselineNetwork()
            else:
                global_network = QAPlannerNetwork(constants.RL_GRU_SIZE, 1, 1)
            global_network.create_net()
        if constants.USE_NAVIGATION_AGENT:
            with tf.variable_scope('nav_global_network') as net_scope:
                free_space_network = FreeSpaceNetwork(constants.GRU_SIZE, 1, 1)
                free_space_network.create_net()
        else:
            net_scope = None

        # prepare session
        sess = tf_util.Session()

        if constants.PREDICT_DEPTH:
            from depth_estimation_network import depth_estimator
            with tf.variable_scope('') as depth_scope:
                depth_estimator = depth_estimator.get_depth_estimator(sess)
        else:
            depth_scope = None

        sess.run(tf.global_variables_initializer())

        # Initialize pretrained weights after init.
        if constants.PREDICT_DEPTH:
            depth_estimator.load_weights()

        #testing_threads = []

        testing_thread = A3CTestingThread(0, sess, net_scope, depth_scope)
        #for i in range(constants.PARALLEL_SIZE):
            #testing_thread = A3CTestingThread(i, sess, net_scope, depth_scope)
            #testing_threads.append(testing_thread)

        tf_util.restore_from_dir(sess, constants.CHECKPOINT_DIR, True)

        if constants.USE_NAVIGATION_AGENT:
            print('now trying to restore nav model')
            tf_util.restore_from_dir(sess, os.path.join(constants.CHECKPOINT_PREFIX, 'navigation'), True)

    sess.graph.finalize()

    question_types = constants.USED_QUESTION_TYPES
    rows = []
    for q_type in question_types:
        curr_rows = list(range(len(testing_thread.agent.game_state.test_datasets[q_type])))
        rows.extend(list(zip(curr_rows, [q_type] * len(curr_rows))))

    #random.seed(999)
    if constants.RANDOM_BY_SCENE:
        rows = shuffle_by_scene(rows)
    else:
        random.shuffle(rows)
    #print (rows)

    answers_correct = []
    ep_lengths = []
    ep_rewards = []
    invalid_percents = []
    #time_lock = threading.Lock()
    if not os.path.exists(constants.LOG_FILE):
        os.makedirs(constants.LOG_FILE)
    out_file = open(constants.LOG_FILE + '/results_' + constants.TEST_SET + '_' + py_util.get_time_str() + '.csv', 'w')
    out_file.write(constants.LOG_FILE + '\n')
    out_file.write('question_type, answer_correct, answer, gt_answer, episode_length, invalid_action_percent, scene number, seed, required_interaction, union, inter, max, early_stop\n')
    #out_file.write('question_type, answer_correct, answer, gt_answer, episode_length, invalid_action_percent, scene number, seed, required_interaction\n')

    qtype = 0
    qobj = 'Apple'
    qcon = 'Fridge'
    qobj_idx = constants.OBJECTS.index(qobj)
    qcon_idx = constants.OBJECTS.index(qcon)

    
    #def test_function(thread_ind):
    #testing_thread = testing_threads[thread_ind]
    sess.run(testing_thread.sync)
    #from game_state import QuestionGameState
    #if testing_thread.agent.game_state is None:
        #testing_thread.agent.game_state = QuestionGameState(sess=sess)
    ###while len(rows) > 0:
    #time_lock.acquire()
    #if len(rows) == 0:
    #   break
    #row = rows.pop()
    if qtype == 2:
        question = ((qobj_idx, qcon_idx), qtype)
    else:
        question = ((qobj_idx), qtype)
    #time_lock.release()

    answer_correct, answer, gt_answer, ep_length, ep_reward, invalid_percent, union, inter, maxc, early_stop = testing_thread.process(question)
    #answer_correct, answer, gt_answer, ep_length, ep_reward, invalid_percent, scene_num, seed, required_interaction, early_stop = testing_thread.process(row)
    question_type = qtype + 1

    #time_lock.acquire()
    output_str = ('%d, %d, %d, %d, %d, %f, %d, %d, %d, %d\n' % (question_type, answer_correct, answer, gt_answer, ep_length, invalid_percent, union, inter, maxc, early_stop))
    #output_str = ('%d, %d, %d, %d, %d, %f, %d, %d, %d, %d\n' % (question_type, answer_correct, answer, gt_answer, ep_length, invalid_percent, scene_num, seed, required_interaction, early_stop))
    out_file.write(output_str)
    out_file.flush()
    answers_correct.append(int(answer_correct))
    ep_lengths.append(ep_length)
    ep_rewards.append(ep_reward)
    invalid_percents.append(invalid_percent)
    print('###############################')
    print('num episodes', len(answers_correct))
    print('average correct', np.mean(answers_correct))
    print('invalid percents', np.mean(invalid_percents), np.median(invalid_percents))
    print('###############################')
    #time_lock.release()

    """
    test_threads = []
    for i in range(constants.PARALLEL_SIZE):
        test_threads.append(threading.Thread(target=test_function, args=(i,)))

    for t in test_threads:
        t.start()

    for t in test_threads:
        t.join()
    """

    out_file.close()

def shuffle_by_scene(rows):
    #test_data
    question_types = ['existence', 'counting', 'contains']
    test_datasets = []
    for qq,question_type in enumerate(question_types):
        prefix = 'questions/'
        path = prefix + 'val/' + constants.TEST_SET + '/data' + '_' + question_type
        #print('path', path)
        data_file = sorted(glob.glob(path + '/*.h5'), key=os.path.getmtime)
        if len(data_file) > 0 and qq in constants.USED_QUESTION_TYPES:
            dataset = h5py.File(data_file[-1])
            dataset_np = dataset['questions/question'][...]
            dataset.close()
            test_dataset = dataset_np
            sums = np.sum(np.abs(test_dataset), axis=1)
            test_datasets.append(test_dataset[sums > 0])
            print('Type', question_type, 'test num_questions', test_datasets[-1].shape)
        else:
            test_datasets.append([])

    rows_np = np.empty((0, 3), int)
    for question_row, question_type_ind in rows:
        scene_num = test_datasets[question_type_ind][question_row, :][0]
        
        #print ("data: ",question_row, question_type_ind,scene_num)
        if scene_num in constants.USED_SCENE:
            rows_np = np.concatenate((rows_np, [[question_row, question_type_ind, scene_num]]))
    
    rows_np = rows_np[rows_np[:,2].argsort()]
    #print (rows_np)

    rows_np_slim = np.array([], int).reshape(0,3)
    for i in np.unique(rows_np[:,2]):
        mask = np.where(rows_np[:,2] == i)
        rows_np[mask] = np.random.permutation(rows_np[mask])
        #print ("rows_np mask: ",rows_np[mask].shape)
        rows_np_slim = np.vstack([rows_np_slim, rows_np[mask][:11, :]])

    #print ("rows: ",rows.shape)
    #print ("rows_np: ",rows_np.shape)
    #print ("rows_np: ",rows_np[:, :2])
    #print ("rows_np_slim: ",rows_np_slim.shape)
    #print ("rows_np_slim: ",rows_np_slim[:, :2])
        

    #return rows
    return list(rows_np[:, :2])
    #return list(rows_np_slim[:, :2])

if __name__ == '__main__':
    main()
