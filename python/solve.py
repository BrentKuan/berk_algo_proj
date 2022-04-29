"""Solves an instance.

Modify this file to implement your own solvers.

For usage, run `python3 solve.py --help`.
"""

import argparse
from ast import Break
from pathlib import Path
from typing import Callable, Dict

from instance import Instance
from solution import Solution
from file_wrappers import StdinFileWrapper, StdoutFileWrapper
import numpy as np
from point import Point

# def calc_distance(x1,x2):
#     sum_vectors = np.sum(np.square(x1 - x2))
#     return np.sqrt(sum_vectors)
    
def solve_naive(instance: Instance) -> Solution:

    num_lines = instance.N
    dim = instance.D
    service_radius = instance.R_s
    rp = instance.R_p
    cities = instance.cities
    affected_area = []
    # Solution 1: Naive Greedy
    for a in range(service_radius+1): # Getting area covered by a tower
        upper = int(((service_radius**2)-(a**2))**0.5)
        for b in range(upper+1):
            affected_area.append([a,b]) 
    
    towers = []
    towers_readable = []

    cover_ratio = np.zeros((dim,dim))
    #naive greedy method
    for row in range(dim):
        for col in range(dim):
            num = 0
            for city in cities:
                if(city.distance_obj(Point(x=row, y=col))<=service_radius):
                    num += 1
            cover_ratio[row][col] = num

    overlap = np.zeros((dim,dim))

    while(len(cities)!=0):
        #greedy step
        cover_ratio_flatten = cover_ratio.flatten()
        max_value = np.max(cover_ratio_flatten)
        maxes = [i for i, j in enumerate(cover_ratio_flatten) if j == max_value]
        temp = np.inf
        for y in maxes:
            row = y//dim
            col = y%dim
            if(overlap[row][col]<temp):
                temp = overlap[row][col]
                max_row = row
                max_col = col
        tower_to_add = Point(x=max_row, y=max_col)
        towers.append(tower_to_add)
        towers_readable.append([max_row,max_col])
        cities_to_remove = []
        for city in cities:
            if(tower_to_add.distance_obj(city)<=service_radius):
                cities_to_remove.append(city)

        # update cover ratio
        for i in range(len(cities_to_remove)):
            cities.remove(cities_to_remove[i])
            for combi in affected_area:
                combi_index_0 = combi[0]
                combi_index_1 = combi[1]
                city_index_0 = cities_to_remove[i].x
                city_index_1 = cities_to_remove[i].y
    
                new_index_1 = [element1 + element2 for (element1, element2) in zip([city_index_0,city_index_1],[combi_index_0,combi_index_1])]
                new_index_2 = [element1 + element2 for (element1, element2) in zip([city_index_0,city_index_1],[-combi_index_0,-combi_index_1])]
                new_index_3 = [element1 + element2 for (element1, element2) in zip([city_index_0,city_index_1],[combi_index_0,-combi_index_1])]
                new_index_4 = [element1 + element2 for (element1, element2) in zip([city_index_0,city_index_1],[-combi_index_0,combi_index_1])]
                new_index = set()
                new_index.add(tuple(new_index_1))
                new_index.add(tuple(new_index_2))
                new_index.add(tuple(new_index_3))
                new_index.add(tuple(new_index_4))
                for index in new_index:
                    row = index[0]
                    col = index[1]
                    if row >=0 and col >=0 and row < dim and col < dim:
                        cover_ratio[row][col] -= 1

        
            for combi in affected_area:
                combi_index_0 = combi[0]
                combi_index_1 = combi[1]
                tower_index_0 = tower_to_add.x
                tower_index_1 = tower_to_add.y
    
                new_index_1 = [element1 + element2 for (element1, element2) in zip([tower_index_0,tower_index_1],[combi_index_0,combi_index_1])]
                new_index_2 = [element1 + element2 for (element1, element2) in zip([tower_index_0,tower_index_1],[-combi_index_0,-combi_index_1])]
                new_index_3 = [element1 + element2 for (element1, element2) in zip([tower_index_0,tower_index_1],[combi_index_0,-combi_index_1])]
                new_index_4 = [element1 + element2 for (element1, element2) in zip([tower_index_0,tower_index_1],[-combi_index_0,combi_index_1])]
                new_index = set()
                new_index.add(tuple(new_index_1))
                new_index.add(tuple(new_index_2))
                new_index.add(tuple(new_index_3))
                new_index.add(tuple(new_index_4))
                for index in new_index:
                    row = index[0]
                    col = index[1]
                    if row >=0 and col >=0 and row < dim and col < dim:
                        overlap[row][col] += 1

        

    print(towers_readable)
    
    return Solution(
        instance=instance,
        towers=towers,
    )


SOLVERS: Dict[str, Callable[[Instance], Solution]] = {
    "naive": solve_naive
}


# You shouldn't need to modify anything below this line.
def infile(args):
    if args.input == "-":
        return StdinFileWrapper()

    return Path(args.input).open("r")


def outfile(args):
    if args.output == "-":
        return StdoutFileWrapper()

    return Path(args.output).open("w")


def main(args):
    with infile(args) as f:
        instance = Instance.parse(f.readlines())
        solver = SOLVERS[args.solver]
        solution = solver(instance)
        print("Penalty is:",solution.penalty())
        assert solution.valid()
        with outfile(args) as g:
            print("# Penalty: ", solution.penalty(), file=g)
            solution.serialize(g)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve a problem instance.")
    parser.add_argument("input", type=str, help="The input instance file to "
                        "read an instance from. Use - for stdin.")
    parser.add_argument("--solver", required=True, type=str,
                        help="The solver type.", choices=SOLVERS.keys())
    parser.add_argument("output", type=str,
                        help="The output file. Use - for stdout.",
                        default="-")
    main(parser.parse_args())
