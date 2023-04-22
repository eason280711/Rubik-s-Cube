import optimal.solver as sv
import optimal.enums as en 
import optimal.face as face
import optimal.cubie as cubie
from tqdm import tqdm

import numpy as np

colormap = {'1': 'U','2': 'R','3': 'F','4': 'D','5': 'L','6': 'B','0':'0','7':'7'}

def to_cube(data):
    return ''.join([colormap[d] for d in data])

def array_to_str(arr):
    return ''.join([str(int(x)) for x in arr.flatten()])

def str_to_array(s):
    arr = np.zeros((6, 3, 3))
    for i in range(len(s)):
        arr[i // 9][(i % 9) // 3][(i % 9) % 3] = int(s[i])
    return arr

def check_is_cube(cube):
    fc = face.FaceCube()
    fc.f = [-1]*54
    fc.from_string(to_cube(cube))
    cc = fc.to_cubie_cube()

    edge_count = [0] * 12
    for i in en.Edge:
        if cc.ep[i] > 0 and cc.ep[i] < 12:
            edge_count[cc.ep[i]] += 1
    for i in en.Edge:
        if edge_count[i] > 1:
            return False

    corner_count = [0] * 8
    for i in en.Corner:
        if cc.cp[i] > 0 and cc.cp[i] < 8:
            corner_count[cc.cp[i]] += 1
    for i in en.Corner:
        if corner_count[i] > 1:
            return False

    return True

def verify(cube):
    fc = face.FaceCube()
    s = fc.from_string(to_cube(cube))
    if s != cubie.CUBE_OK:
        return False

    cc = fc.to_cubie_cube()
    s = cc.verify()

    if s != cubie.CUBE_OK:
        return False

    return True

n = 0

def exhaustive_search(a, b, i, j, k, fastest_solution=None):
    global n
    
    if i == 6:
        if verify(array_to_str(a)):
            if fastest_solution is None or n < fastest_solution[0]:
                fastest_solution = (n, a.copy())
        return fastest_solution
        
    if j == 3:
        return exhaustive_search(a, b, i + 1, 0, k, fastest_solution)
    if k == 3:
        return exhaustive_search(a, b, i, j + 1, 0, fastest_solution)
    
    if abs(round(b[i][j][k]) - b[i][j][k]) <= 0.2:
        a[i][j][k] = round(b[i][j][k])
        if check_is_cube(array_to_str(a)):
            fastest_solution = exhaustive_search(a.copy(), b, i, j, k + 1, fastest_solution)
    else:
        a[i][j][k] = round(b[i][j][k]+0.5)
        if check_is_cube(array_to_str(a)):
            fastest_solution = exhaustive_search(a.copy(), b, i, j, k + 1, fastest_solution)
        
        a[i][j][k] = round(b[i][j][k]-0.5)
        if check_is_cube(array_to_str(a)):
            fastest_solution = exhaustive_search(a.copy(), b, i, j, k + 1, fastest_solution)
    
    n += 1
    return fastest_solution

test_data_tmp = np.array([
    [
        [0.9, 0.9, 0.9],
        [0.9, 0.9, 0.9],
        [0.9, 0.9, 0.9]
    ],
    [
        [1.9, 1.9, 1.9],
        [1.9,1.9, 1.9],
        [1.9,1.9, 1.9]
    ],
    [
        [ 3,3, 3],
        [3, 3, 3],
        [3, 3,3]
    ],
    [
        [4, 4, 4],
        [4, 4,4],
        [4, 4,4]
    ],
    [
        [5, 5,5],
        [5, 5, 5],
        [5, 5, 5]
    ],
    [
        [6, 6, 6],
        [6, 6, 6],
        [6, 6, 6]
    ]
])

test_data = np.array([
    [
        [1.8,0.8,1.4],
        [2.6,0.9,1.8],
        [2.8,0.8,1.9]
    ],
    [
        [1.8,0.8,2.7],
        [2.2,1.8,1.5],
        [3.6,2.2,2.1]
    ],
    [
        [4.1,3.6,2.8],
        [3.6,2.8,2],
        [3.1,3.2,2.5]
    ],
    [
        [4,3.9,3.7],
        [4,3.8,3.6],
        [4.2,3.2,2.6]
    ],
    [
        [3.9,5.3,4.3],
        [3.7,4.8,4.7],
        [4.7,5.2,3.8]
    ],
    [
        [5.2,5.1,3.1],
        [6,6,4.1],
        [6,5.6,3.1]
    ]
])


result = exhaustive_search(np.zeros((6, 3, 3)), test_data, 0, 0, 0)

print('---------------------')
print('Total number of cases: ', n)

print('---------------------')
print(result)

print('---------------------')
print(len(result))

print(array_to_str(test_data))