import sys

sys.stdout = open('NUL', 'w')
import os
import pycuber as pc
from pycuber.solver import CFOPSolver

import optimal.solver as sv
import random
import json
import sys

from tqdm import tqdm, trange

import numpy as np

from timeout import timeout_function, slow_function
import time
from concurrent.futures import ThreadPoolExecutor

sys.stdout = sys.__stdout__

colorDict = {"[w]": "D", "[y]": "U", "[o]": "R", "[r]": "L", "[g]": "F", "[b]": "B"}

def add_content_to_json(array1, array2, array3, distance1, distance2, filename):
    #with open(filename, 'r') as f:
    #    data = json.load(f)

    # 將輸入的數據轉換為JSON格式
    new_data = {
        "initial_state": array1,
        "destination_state_1": array2,
        "destination_state_2": array3,
        "distance_1": distance1,
        "distance_2": distance2
    }

    with open(filename, 'w') as f:
        json.dump(new_data, f)

def cubeAsArray(cube):
    faces = ["U", "R", "F", "D", "L", "B"]
    cubeArray = []
    m_str = str()
    for face in faces:
        face = cube.get_face(face)
        subFace = []
        for x in [0,1,2]:
            subLine = []
            for y in [0,1,2]:
                subLine.append(colorDict[str(face[x][y])])
                m_str += colorDict[str(face[x][y])]
            subFace.append(subLine)
        cubeArray.append(subFace)
    return cubeArray, m_str

def generate_random_moves(N):
    moves = ['F', 'F\'', 'F2', 'B', 'B\'', 'B2', 'L', 'L\'', 'L2', 'R', 'R\'', 'R2', 'U', 'U\'', 'U2', 'D', 'D\'', 'D2']
    #print("Random: " + str(N))
    random_moves = []
    for i in range(N):
        move = random.choice(moves)
        random_moves.append(move)
    return " ".join(random_moves)

def calculate_ratio(nums_dict):
    total_count = sum(nums_dict.values())
    ratios = {}
    for num, count in nums_dict.items():
        ratios[num] = count / total_count
    return ratios

def Gen_cube(c,b):

    sys.stdout = open('NUL', 'w')
    Solution = sv.solveto(c, b)
    sys.stdout = sys.__stdout__

    return len(Solution.split(" "))-1

def randcube(input_tuple):
    N,progress_bar = input_tuple
    Mode = ["Origin","Random"]
    mode = random.choice(Mode)
    c = pc.Cube()
    alg = pc.Formula()
    random_alg = alg.random()
    if mode == "Random":
        c(random_alg)

    b = pc.Cube()
    if mode == "Random":
        b(random_alg)
    b(generate_random_moves(random.randint(10,25)))

    d = pc.Cube()
    if mode == "Random":
        d(random_alg)
    d(generate_random_moves(random.randint(1,9)))

    c_dic, c_str = cubeAsArray(c)
    b_dic, b_str = cubeAsArray(b)
    d_dic, d_str = cubeAsArray(d)

    result1 = timeout_function(Gen_cube,args=(c_str,b_str), timeout=30, default=18)
    result2 = timeout_function(Gen_cube,args=(c_str,d_str), timeout=30, default=18)

    if result2 > result1:
        result1, result2 = result2, result1
        b_dic, d_dic = d_dic, b_dic
    
    add_content_to_json(c_dic, b_dic, d_dic, result1, result2, "datatest2/"+str(N)+".json")
    progress_bar.update(1)


if __name__ == '__main__':
    progress_bar = tqdm(total=int(1000),ascii=True,desc="Generating dataset")
    inputs = [(i,progress_bar) for i in range(int(1000))]
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(randcube, inputs)