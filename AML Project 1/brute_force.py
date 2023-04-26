import itertools
import numpy as np
import random

def get_theta(points, labels):
    theta_labels_dict = {}
    search = [round(x, 2) for x in np.linspace(-20, 20, 401).tolist()]
    for theta in search:
        label_sublist = []
        for point in points:
            label = 0
            if (point >= theta and point <= round(theta + 1, 2)) or (point >= round(theta + 2, 2) and point <= round(theta + 4, 2)) or (point >= round(theta + 6, 2) and point <= round(theta + 9, 2)):
                label = 1
            label_sublist.append(label)
        theta_labels_dict[theta] = label_sublist
    
    for key in theta_labels_dict.keys():
        if theta_labels_dict[key] == labels:
            return key
    
    return 'not found'

def brute_force():
    points_list = []
    alphas = np.linspace(-10, 10, 100)
    for alpha in alphas:
        for i in range(-10, 10):
            points_sublist = []
            for j in range(4):
                points_sublist.append(i + j + alpha)
            points_list.append(points_sublist)

    labels = list(itertools.product([0, 1], repeat=4))
    for points in points_list:
        print(f'Trying points {points}') 
        theta_list = []
        for labelling in labels:
            res = get_theta(points, list(labelling))
            if res != 'not found':
                theta_list.append(res)
        
        if len(theta_list) == 16:
            print(f'Found points {points} with thetas {theta_list}')
            break

def brute_force_2():
    labels = list(itertools.product([0, 1], repeat=5))
    search_grid = list(round(x, 2) for x in np.linspace(-10, 10, 201).tolist())
    for idx,c1 in enumerate(search_grid):
        for c2 in search_grid[idx+1:]:
            if c2 <= c1:
                continue
            for c3 in search_grid[idx+2:]:
                if c3 <= c2:
                    continue
                for c4 in search_grid[idx+3:]:
                    if c4 <= c3: 
                        continue
                    for c5 in search_grid[idx+4:]:
                        if c5 <= c4: 
                            continue
                        points = [c1, c2, c3, c4, c5]
                        print(f'Trying points ({c1}, {c2}, {c3}, {c4}, {c5})')
                        theta_list = []
                        for labelling in labels:
                            res = get_theta(points, list(labelling))
                            if res != 'not found':
                                theta_list.append(res)
                        
                        print(len(theta_list))
                        if len(theta_list) == 32:
                            print(f'Found points {points} with thetas {theta_list}')
                            break
# brute_force()
# brute_force_2()
points = [-5, -4.2, -1.8, 1.6]
labels = list(itertools.product([0, 1], repeat=4))
for idx, labelling in enumerate(labels):
    res = get_theta(points, list(labelling))
    # print(f'{idx + 1}: Labelling {labelling}: [{res}, {round(res + 1, 2)}], [{round(res + 2, 2)}, {round(res + 4, 2)}], [{round(res + 6, 2)}, {round(res + 9, 2)}]')
    # print(f'\\item For $L={labelling}$, take $\\theta_{idx + 1}={(res)}$,')
    # print(f'\mathbf 1_[{res}, {round(res + 1, 2)}], [{round(res + 2, 2)}, {round(res + 4, 2)}], [{round(res + 6, 2)}, {round(res + 9, 2)}]')
    print(res)