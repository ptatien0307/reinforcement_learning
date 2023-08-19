from ortools.algorithms import pywrapknapsack_solver
import re
import os
import time

def read_file(path):
    with open(path, 'rb') as f:
        x = [re.sub('\r|\n', '', line.decode("utf-8"))  for line in f.readlines()]
        x = [i for i in x if i != '']
        n = int(x[0])
        c = int(x[1])
        values = []
        weights = [[]]
        for item in x[2:]:
            x, y = item.split(' ')
            values.append(int(x))
            weights[0].append(int(y))
        return n, c, values, weights


def main(n, c, values, weights):
    # Create the solver.
    solver = pywrapknapsack_solver.KnapsackSolver(
        pywrapknapsack_solver.KnapsackSolver.
        KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, 'KnapsackExample')

    capacities = [c]

    solver.Init(values, weights, capacities)
    solver.set_time_limit(300)

    start = time.time()
    computed_value = solver.Solve()
    end = time.time()

    packed_items = []
    packed_weights = []
    total_weight = 0
    print('Total value =', computed_value)
    for i in range(n):
        if solver.BestSolutionContains(i):
            packed_items.append(i)
            packed_weights.append(weights[0][i])
            total_weight += weights[0][i]
    print('Total weight:', total_weight)
    print('Time: ', end - start)
    # print('Packed items:', packed_items)
    # print('Packed_weights:', packed_weights)


if __name__ == '__main__':
    test_case = ['00Uncorrelated', '01WeaklyCorrelated', '02StronglyCorrelated', '03InverseStronglyCorrelated', '04AlmostStronglyCorrelated',
                 '05SubsetSum', '06UncorrelatedWithSimilarWeights', '07SpannerUncorrelated', '08SpannerWeaklyCorrelated', '09SpannerStronglyCorrelated',
                 '10MultipleStronglyCorrelated', '11ProfitCeiling', '12Circle']
    
    numbers = ['n00050', 'n00200', 'n01000', 'n05000', 'n10000']
    a = 'D:\CS106\BT2\kplib'
    b = 'R01000\s000.kp'
    print(f'{test_case[9]}')
    for j in numbers:
        print(f'n: {j} -----')
        path = os.path.join(a, test_case[9], j, b)
        n, c, values, weights = read_file(path)
        main(n, c, values, weights)