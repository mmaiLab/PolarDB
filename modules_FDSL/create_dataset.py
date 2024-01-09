import numpy as np
import pickle as pkl
import argparse
import random
import math


def parse_args():
    parser = argparse.ArgumentParser(description='demo')

    parser.add_argument('--save_name', type=str, help='output file path', default='PolarDB.pkl')
    parser.add_argument("--spatial_noise", action='store_true')
    parser.add_argument("--temporal_noise", action='store_true')

    parser.add_argument('--used_rate', type=int, default=100)

    args = parser.parse_args()
    return args


def round_external_coordinates(x, max_x, min_x):
    # x is larger than screen bounds.
    if x > max_x:
        return -1
    
    # x is smaller than screen bounds.
    if x < 0:
        return -1
    
    # x is in screen bounds
    return x


"""
    PolarDB two dimension polar equations
"""
def traj_archimedesspiral_curve(a, ts):
    traj = []
    for t in ts:
        x = a * t * np.cos(t)
        y = a * t * np.sin(t)
        traj.append([x, y])
    return traj


def traj_cardioid_curve(a, ts):
    traj = []
    for t in ts:
        x = a * (1 + np.cos(t)) * np.cos(t)
        y = a * (1 + np.cos(t)) * np.sin(t)
        traj.append([x, y])
    return traj


def traj_asteroid_curve(a, ts):
    traj = []
    for t in ts:
        x = a * (np.cos(t) * np.cos(t) * np.cos(t))
        y = a * (np.sin(t) * np.sin(t) * np.sin(t))
        traj.append([x, y])
    return traj


def traj_ellipse(a, b, ts):
    traj = []
    for t in ts:
        x = a * np.cos(t)
        y = b * np.sin(t)
        traj.append([x, y])
    return traj


def traj_ellipse_mix(a, b, ts):
    traj = []
    half_l = int(len(ts)/2)
    for i, t in enumerate(ts):
        if i < half_l:
            x = a * np.cos(t)
            y = b * np.sin(t)
        else:
            x = b * np.cos(t)
            y = a * np.sin(t)       
        traj.append([x, y])
    return traj


def traj_point(ts):
    traj = []
    for t in ts:
        traj.append([0, 0])
    return traj


def traj_lissajous(a, b, p, q, ts):
    traj = []
    for t in ts:
        x = a * np.cos(p*t)
        y = b * np.sin(q*t + (math.pi/2))
        traj.append([x, y])
    return traj


def traj_custom_001(w, h):
    # sinusoidal wave
    traj = []
    for t in ts:
        x = (w / len(ts)) * t
        y = h * np.sin(t)
        traj.append([x, y])
    return traj

if __name__ == '__main__':
    args = parse_args()
    print('Creating PolarDB. Save name: {}'.format(args.save_name))

    max_h = 1440
    max_w = 1920
    unit_h = max_h/8
    unit_w = max_w/8

    dataset = {}
    label = {}
    stat = {}
    dataset_metrics = {'total_length': 0, 'data_num': 0, 'used': 0, 'unused': 0}
    for max_t in [1, 3, 5, 10, 15, 20, 30]:
        print('max_t={}'.format(max_t))

        # Class label for length parameter
        if max_t <= 5:
            length_class = 'short'
        elif 5 < max_t <= 10:
            length_class = 'normal'
        else:
            length_class = 'long'

        for span_t in [0.1, 0.3, 0.8]: # rotation speed parameter
            ts = np.arange(0, 6.4*max_t, span_t)

            # list of instances
            listed = {}

            # (1) point
            listed['point'] = traj_point(ts)

            # (2, 3) cardioid and asteroid
            for a in [10, 75, 150, 300]:
                listed['cardioid_{}'.format(a)] = traj_cardioid_curve(a, ts)
                listed['asteroid_{}'.format(a)] = traj_asteroid_curve(a, ts)

            # (4, 5) ellipse and circle
            for a in [10, 75, 300]:
                for b in [10, 75, 300]:
                    if a == b:
                        listed['circle_{}_{}'.format(a, b)] = traj_ellipse(a, b, ts)
                    else:
                        if a > b:
                            listed['ellipse@horizontal_{}_{}'.format(a, b)] = traj_ellipse(a, b, ts)
                        elif a < b:
                            listed['ellipse@vertial_{}_{}'.format(a, b)] = traj_ellipse(a, b, ts)
            
            # ellipse mix
            for (a, b) in [(10, 100), (200, 10)]:
                listed['ellipse@mix_{}_{}'.format(a, b)] = traj_ellipse_mix(a, b, ts)

            # lissajous
            for a in [75, 150, 300]:
                listed['lissajous@1:2_{}_{}'.format(a, a)] = traj_lissajous(a, a, 1, 2, ts)
                listed['lissajous@3:2_{}_{}'.format(a, a)] = traj_lissajous(a, a, 3, 2, ts)
                listed['lissajous@3:4_{}_{}'.format(a, a)] = traj_lissajous(a, a, 3, 4, ts)
        

            # sinusoidal wave
            for (w, h) in [(500, 300), (500, 150), (300, 300), (300, 150)]:
                listed['costom001_{}_{}'.format(w, h)] = traj_custom_001(w, h)
            
            # (6) archimedes spiral
            for a in [5, 10]:
                listed['archimedesspiral_{}'.format(a)] = traj_archimedesspiral_curve(a, ts)

            # Division into right and left hands
            for x_l in [unit_w, unit_w*2, unit_w*3]:
                for y_l in [unit_h, unit_h*4, unit_h*7]:
                    for x_r in [max_w/2 + unit_w, max_w/2 + unit_w*2, max_w/2 + unit_w*3]:
                        for y_r in [unit_h, unit_h*4, unit_h*7]:
                            for k_r, v_r in listed.items():
                                for k_l, v_l in listed.items():
                                    
                                    # random sampling
                                    rn = random.randint(1, 100)
                                    if args.used_rate >= rn:
                                        dataset_metrics['used'] += 1
                                    else:
                                        dataset_metrics['unused'] += 1
                                        continue

                                    name = 'traj_{}_{}_{}_{}_{}_{}_{}_{}'.format(k_r, k_l, max_t, span_t, x_l, y_l, x_r, x_l)
                                    class_label = '{}+{}+{}'.format(k_l.split('_')[0], k_r.split('_')[0], length_class)
                                    new_v_l, new_v_r = [], []
                                    noi_x, noi_y = 0, 0

                                    for i in range(len(v_l)):
                                        if args.spatial_noise:
                                            noi_x = random.randint(-7, 7)
                                            noi_y = random.randint(-7, 7)
                                        new_v_l.append([
                                            round_external_coordinates(int(v_l[i][0] + x_l + noi_x), max_w, 0),
                                            round_external_coordinates(int(v_l[i][1] + y_l + noi_y), max_h, 0)
                                        ])

                                    for i in range(len(v_r)):
                                        if args.spatial_noise:
                                            noi_x = random.randint(-7, 7)
                                            noi_y = random.randint(-7, 7)
                                        new_v_r.append([
                                            round_external_coordinates(int(v_r[i][0] + x_r + noi_x), max_w, 0),
                                            round_external_coordinates(int(v_r[i][1] + y_r + noi_y), max_h, 0)
                                        ])

                                    dataset[name] = [new_v_l, new_v_r]
                                    label[name] = class_label
                                    stat[class_label] = stat[class_label]+1 if class_label in stat else 1

                                    # metrics
                                    dataset_metrics['total_length'] += len(new_v_r)
                                    dataset_metrics['data_num'] += 1
            
    print('===== dataset information =====')
    print(len(dataset))
    print('used: {}, unused: {}'.format(dataset_metrics['used'], dataset_metrics['unused']))
    print('avg length = {}'.format(dataset_metrics['total_length'] / dataset_metrics['data_num']))
    print('===== class statistics =====')
    class_num = 0
    for k, v in stat.items():
        class_num += 1
        print('{}: {}'.format(k, v))
    print('class_num = {}'.format(class_num))
    
    if args.savename is not None:
        with open(args.savename, 'wb') as f:
            pkl.dump({'dataset': dataset, 'label': label, 'stat': stat}, f, protocol=4)
    else:
        print('Not save dataset.')
