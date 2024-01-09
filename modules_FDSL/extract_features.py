import os
import numpy as np
import glob
import time
import argparse
import math
import pickle as pkl
import pandas as pd
import random
from scipy.spatial import distance


def parse_args():
    parser = argparse.ArgumentParser(description='demo')

    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)

    # modification options
    parser.add_argument("--relative", action='store_true')
    parser.add_argument("--smooth", action='store_true')
    parser.add_argument("--normalize2x", action='store_true')
    parser.add_argument("--normalize1x", action='store_true')

    # feature type options
    parser.add_argument("--velocity", action='store_true')
    parser.add_argument("--acceleration", action='store_true')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print('Loading..')
    with open(args.dataset_path, 'rb') as f:
        dataset_data = pkl.load(f)
        dataset = dataset_data['dataset']
        label   = dataset_data['label']
        stat    = dataset_data['stat']
    print('Loading has completed!')
    print('dataset length = {}'.format(len(dataset)))

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if not os.path.exists(os.path.join(args.output_path, 'data')):
        os.makedirs(os.path.join(args.output_path, 'data'))
    print('Making class data..')
    with open(os.path.join(args.output_path, 'class.txt'), 'w') as f2:
        fdsl_trajectory_class_to_id = {}
        fdsl_trajectory_id_to_class = {}
        i = 0
        for k, v in stat.items():
            fdsl_trajectory_class_to_id[k] = i 
            fdsl_trajectory_id_to_class[i] = k
            class_path = os.path.join(args.output_path, 'data', k)
            if not os.path.exists(class_path):
                os.makedirs(class_path)
            f2.write('{}\n'.format(','.join([str(i), k])))
            i += 1
    print('Making class data has completed!')


    bbox_size = 200 
    print('dataset length is {}'.format(len(dataset)))
    count = 0
    num_train, num_val, num_test = 0, 0, 0
    metadatas = {}
    invalids = []
    for k, v in dataset.items():
        count += 1
        if count % 50 == 0:
            print('Processing {}.'.format(count))
        hod_data = {}
        length = len(v[0])
        if len(v[0]) != len(v[1]):
            print('Do not match hand trajectory length between left and right.')
            invalids.append(k)
            continue

        # (1) Bounding boxes
        hod_data['hand_L'], hod_data['hand_R'] = [], []
        for i in range(length):
            hod_data['hand_L'].append([v[0][i][0]-100, v[0][i][1]+100,
                                       v[0][i][0]+100, v[0][i][1]-100,
                                       v[0][i][0], v[0][i][1]])
            hod_data['hand_R'].append([v[1][i][0]-100, v[1][i][1]+100,
                                       v[1][i][0]+100, v[1][i][1]-100,
                                       v[1][i][0], v[1][i][1]])


        flag = random.randint(1, 10)
        if flag in [1, 2]:
            hod_data['mode'] = 'test'
            num_test += 1
        elif flag in [3, 4]:
            hod_data['mode'] = 'val'
            num_val += 1
        else:
            hod_data['mode'] = 'train'
            num_train += 1

        metadatas[k] = {'mode': hod_data['mode'], 
                        'len': str(len(hod_data['hand_L']))}


        if os.path.exists(os.path.join(args.output_path, 'data', label[k], '{}.pkl'.format(k))):
            continue
        
        
        # (2) position
        hod_data['org_hand_L'], hod_data['org_hand_R'] = hod_data['hand_L'], hod_data['hand_R']

        # (3) distance
        hod_data['hand_distance'] = []
        for i in range(length):
                hand_L = hod_data['hand_L'][i]
                hand_R = hod_data['hand_R'][i]
                if not (hand_L[4] == -1 or hand_L[5] == -1 or hand_R[4] == -1 or hand_R[5] == -1):
                    hod_data['hand_distance'].append([
                        hand_L[4] - hand_R[4],
                        hand_L[5] - hand_R[5],
                        distance.euclidean((hand_L[4], hand_L[5]),(hand_R[4], hand_R[5]))
                    ])
                else:
                    hod_data['hand_distance'].append([0, 0, 0])

        # (4) relative position
        if args.relative:
            cnt_L, cnt_R = 0, 0
            sum_L, sum_R = [0, 0], [0, 0] # x, y
            for i in range(length):
                hand_L = hod_data['hand_L'][i]
                hand_R = hod_data['hand_R'][i]
                if not (hand_L[4] == -1 or hand_L[5] == -1):
                    cnt_L += 1
                    sum_L[0] += hand_L[4]
                    sum_L[1] += hand_L[5]
                if not (hand_R[4] == -1 or hand_R[5] == -1):
                    cnt_R += 1
                    sum_R[0] += hand_R[4]
                    sum_R[1] += hand_R[5]
            if cnt_L == 0:
                print('hand_L detection result in iter_{} is empty'.format(count))
                mean_L = [0, 0]
            else:
                mean_L = [sum_L[0] / cnt_L, sum_L[1] / cnt_L]
            if cnt_R == 0:
                print('hand_R detection result in iter_{} is empty'.format(count))
                mean_R = [0, 0]
            else:
                mean_R = [sum_R[0] / cnt_R, sum_R[1] / cnt_R]

            for i in range(length):
                hand_L = hod_data['hand_L'][i]
                hand_R = hod_data['hand_R'][i]
                if not (hand_L[4] == -1 or hand_L[5] == -1):
                    hod_data['hand_L'][i][0] = hod_data['hand_L'][i][0] - mean_L[0]
                    hod_data['hand_L'][i][1] = hod_data['hand_L'][i][1] - mean_L[1]
                    hod_data['hand_L'][i][2] = hod_data['hand_L'][i][2] - mean_L[0]
                    hod_data['hand_L'][i][3] = hod_data['hand_L'][i][3] - mean_L[1]
                    hod_data['hand_L'][i][4] = hod_data['hand_L'][i][4] - mean_L[0]
                    hod_data['hand_L'][i][5] = hod_data['hand_L'][i][5] - mean_L[1]
                if not (hand_R[4] == -1 or hand_R[5] == -1):
                    hod_data['hand_R'][i][0] = hod_data['hand_R'][i][0] - mean_R[0]
                    hod_data['hand_R'][i][1] = hod_data['hand_R'][i][1] - mean_R[1]
                    hod_data['hand_R'][i][2] = hod_data['hand_R'][i][2] - mean_R[0]
                    hod_data['hand_R'][i][3] = hod_data['hand_R'][i][3] - mean_R[1]
                    hod_data['hand_R'][i][4] = hod_data['hand_R'][i][4] - mean_R[0]
                    hod_data['hand_R'][i][5] = hod_data['hand_R'][i][5] - mean_R[1]
        
        # (5) normalization
        if args.normalize2x or args.normalize1x:
            print('Not implemented.')
            exit()
        
        # (6) velocity
        if args.velocity:
            for lr in ['hand_L', 'hand_R']:
                hod_data['{}_velocity'.format(lr)] = []
                for i in range(length-1):
                    hand = hod_data[lr][i]
                    next_hand = hod_data[lr][i+1]
                    if not (hand[4] == -1 or hand[5] == -1 or next_hand[4] == -1 or next_hand[5] == -1):
                        hod_data['{}_velocity'.format(lr)].append([next_hand[4]-hand[4], next_hand[5]-hand[5]])
                    else:
                        hod_data['{}_velocity'.format(lr)].append([0, 0])
                hod_data['{}_velocity'.format(lr)].append([0, 0])

            # smooth
            if args.smooth:
                hod_data['hand_L_velocity_smooth'] = np.array(hod_data['hand_L_velocity'])
                hod_data['hand_R_velocity_smooth'] = np.array(hod_data['hand_R_velocity'])

                for key in ['hand_L_velocity_smooth', 'hand_R_velocity_smooth']:
                    if hod_data[key].shape[0] <= convolve_width:
                        continue
                    for i in range(2):
                        hod_data[key][:, i] = valid_convolve(hod_data[key][:, i], convolve_width)
                        hod_data[key][:, i] = valid_convolve(hod_data[key][:, i], convolve_width)

                hod_data['hand_L_velocity_smooth'] = hod_data['hand_L_velocity_smooth'].tolist()
                hod_data['hand_R_velocity_smooth'] = hod_data['hand_R_velocity_smooth'].tolist()

                hod_data['hand_L_velocity'] = hod_data['hand_L_velocity_smooth']
                hod_data['hand_R_velocity'] = hod_data['hand_R_velocity_smooth']
        
        # (6-2) velocity (sparse)
        if True:
            every_n = 10
            for lr in ['hand_L', 'hand_R']:
                hod_data['{}_velocity_every_{}'.format(lr, every_n)] = []
                for i in range(length):
                    if i % every_n == 0:
                        next_index = length-1 if (i+10 >= length-1) else i+10
                        hand = hod_data[lr][i]
                        next_hand = hod_data[lr][next_index]
                        if not (hand[4] == -1 or hand[5] == -1 or next_hand[4] == -1 or next_hand[5] == -1):
                            velocity = [next_hand[4]-hand[4], next_hand[5]-hand[5]]
                        else:
                            velocity = [0, 0]
                    hod_data['{}_velocity_every_{}'.format(lr, every_n)].append(velocity)
        
        # (7) acceleration
        if args.acceleration:
            for lr in ['hand_L', 'hand_R']:
                hod_data['{}_acceleration'.format(lr)] = []
                for i in range(length-1):
                    if args.smooth:
                        hand = hod_data['{}_velocity_smooth'.format(lr)][i]
                        next_hand = hod_data['{}_velocity_smooth'.format(lr)][i+1]
                    else:
                        hand = hod_data['{}_velocity'.format(lr)][i]
                        next_hand = hod_data['{}_velocity'.format(lr)][i+1]
                    if not (hand[0] == -1 or hand[1] == -1 or next_hand[0] == -1 or next_hand[1] == -1):
                        hod_data['{}_acceleration'.format(lr)].append([next_hand[0]-hand[0], next_hand[1]-hand[1]])
                    else:
                        hod_data['{}_acceleration'.format(lr)].append([0, 0])
                hod_data['{}_acceleration'.format(lr)].append([0, 0]) # 最後はパディングで埋める
        
        # ============================================================
        # (B)save
        with open(os.path.join(args.output_path, 'data', label[k], '{}.pkl'.format(k)), 'wb') as f4:
            pkl.dump(hod_data, f4, protocol=4)

    # 4 save meta data
    print('Processing meta data..')
    with open(os.path.join(args.output_path, 'metadata.txt'), 'w') as f3:
        for k, _ in dataset.items():
            if k not in invalids:
                verb_id = fdsl_trajectory_class_to_id[label[k]]
                f3.write('{}\n'.format(','.join([k, metadatas[k]['mode'], metadatas[k]['len'], str(verb_id)])))
    print('Processing meta data has completed!')

    # 5 output
    print(num_train, num_val, num_test)

