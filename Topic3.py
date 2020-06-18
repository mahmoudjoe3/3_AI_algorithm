from builtins import len

import numpy as np
import operator
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from random import seed
from random import randrange
from csv import reader
from math import sqrt
from collections import Counter
import random
r = random.Random()
r.seed("AI")


import math


# region SearchAlgorithms
class Stack:

    def __init__(self):
        self.stack = []

    def push(self, value):
        if value not in self.stack:
            self.stack.append(value)
            return True
        else:
            return False

    def exists(self, value):
        if value not in self.stack:
            return True
        else:
            return False

    def pop(self):
        if len(self.stack) <= 0:
            return ("The Stack == empty")
        else:
            return self.stack.pop()

    def top(self):
        return self.stack[0]


class Node:
    id = None
    up = None
    down = None
    left = None
    right = None
    previousNode = None
    edgeCost = None
    gOfN = None  # total edge cost
    hOfN = None  # heuristic value
    heuristicFn = None

    def __init__(self, value):
        self.value = value
        self.previousNode = None
        self.gOfN = 0
        self.hOfN = 0
        self.id=0
        self.heuristicFn = 0




class SearchAlgorithms:
    path = []  # Represents the correct path from start node to the goal node.
    fullPath = []  # Represents all visited nodes from the start node to the goal node.
    totalCost = None
    #my attribute
    mazeMap = {}
    costMap = {}
    start = None
    end = None
    iid=0
    def __init__(self, mazeStr, edgeCost):
        i = 0
        j = 0
        ii = 0
        while (ii < len(mazeStr)):
            if mazeStr[ii] == ' ':
                j += 1
                i = 0
                ii += 1
                continue
            if mazeStr[ii] == 'S':
                self.start = (j,i)
            elif mazeStr[ii] == 'E':
                self.end = (j,i)
            if mazeStr[ii] != ',':
                self.mazeMap[(j,i)] = mazeStr[ii]
                i += 1
            ii += 1
        iii = 0
        l = 0
        m = 0
        while (iii < len(edgeCost)):
            self.costMap[(m, l)] = edgeCost[iii]
            l += 1
            iii += 1
            if (iii % 7 == 0):
                l = 0
                m += 1

        #now we can deal with mazeMap, costMap, start, end
    def AstarManhattanHeuristic(self):
        node_open_list = []
        node_close_list = []
        start_node = Node(self.start)
        goal_node = Node(self.end)
        node_open_list.append(start_node)

        while len(node_open_list) > 0 :
            node_open_list.sort(key=operator.attrgetter('heuristicFn'))
            current_node = node_open_list.pop(0)
            current_node.id = (current_node.value[0] * 7) + current_node.value[1];
            self.fullPath.append(current_node.id)
            node_close_list.append(current_node)
            # Check if we have reached the goal, return the path
            if current_node.value == goal_node.value:
                path = []
                self.totalCost=0
                while current_node != start_node:
                    path.append(current_node.id)
                    self.totalCost+=self.costMap[current_node.value]
                    current_node = current_node.previousNode
                path.append(0);
                return self.fullPath,path[::-1],self.totalCost

            (x, y) = current_node.value
            # Get neighbors
            neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
            for next in neighbors:
                if next[0]<0 or next[1]<0 or next[0]>=5 or next[1]>=7:
                    neighbors.remove(next)
            # Loop neighbors
            for next in neighbors:
                map_value = self.mazeMap.get(next)
                if (map_value == '#'):
                    continue
                neighbor = Node(next)
                neighbor.previousNode=current_node
                if (neighbor in node_close_list):
                    continue
                neighbor.gOfN=0
                neighbor.hOfN=0
                neighbor.heuristicFn=0
                neighbor.gOfN =current_node.gOfN+self.costMap[neighbor.value]
                neighbor.hOfN = abs(neighbor.value[0] - goal_node.value[0]) + abs(
                    neighbor.value[1] - goal_node.value[1])
                neighbor.heuristicFn = neighbor.gOfN + neighbor.hOfN
                if (self.not_in_open(node_open_list, neighbor) == True):
                    e =0
                    h=0
                    for node in node_open_list:
                        if neighbor.value == node.value:
                            node_open_list[h]=neighbor
                            e=1
                        h+=1
                    if(e==0):
                        node_open_list.append(neighbor)
        return None
    #end AstarManhattanHeuristic
    def not_in_open(self, open, neighbor):
        for node in open:
            if (neighbor.value == node.value and neighbor.heuristicFn >= node.heuristicFn):
                return False
        return True

# endregion

# region KNN
class KNN_Algorithm:
    def __init__(self, K):
        self.K = K

    def euclidean_distance(self, p1, p2):
        DIST = 0.0
        for i in range(len(p1) - 1):
            DIST += (p1[i] - p2[i]) ** 2
        return sqrt(DIST)

    def fit_dataSet(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict_neighbors(self, X):
        predictions_Neighbors = [self._predict(x) for x in X]
        return np.array(predictions_Neighbors)

    def _predict(self, x):
        neighborhood = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indexes = np.argsort(neighborhood)[:self.K]
        output = [self.y_train[i] for i in k_indexes]
        correct_output = Counter(output).most_common(1)
        return correct_output[0][0]

    def calculate_KNN_accuracy(self, exact_y, predected_y):
        acc = (np.sum(exact_y == predected_y) / len(exact_y))*100
        return acc

    def KNN(self, X_train, X_test, Y_train, Y_test):
        self.fit_dataSet(X_train, Y_train)
        predictions_Neighbors = self.predict_neighbors(X_test)
        return self.calculate_KNN_accuracy(Y_test, predictions_Neighbors)
# endregion


# region GeneticAlgorithm
class GeneticAlgorithm:
    Cities = [1, 2, 3, 4, 5, 6]
    DNA_SIZE = len(Cities)
    POP_SIZE = 20
    GENERATIONS = 5000

    """
    - Chooses a random element from items, where items is a list of tuples in
       the form (item, weight).
    - weight determines the probability of choosing its respective item. 
     """

    def weighted_choice(self, items):
        weight_total = sum((item[1] for item in items))
        n = r.uniform(0, weight_total)
        for item, weight in items:
            if n < weight:
                return item
            n = n - weight
        return item

    """ 
      Return a random character between ASCII 32 and 126 (i.e. spaces, symbols, 
       letters, and digits). All characters returned will be nicely printable. 
    """

    def random_char(self):
        return chr(int(r.randrange(32, 126, 1)))

    """ 
       Return a list of POP_SIZE individuals, each randomly generated via iterating 
       DNA_SIZE times to generate a string of random characters with random_char(). 
    """

    def random_population(self):
        pop = []
        for i in range(1, 21):
            x = r.sample(self.Cities, len(self.Cities))
            if x not in pop:
                pop.append(x)
        return pop

    """ 
      For each gene in the DNA, this function calculates the difference between 
      it and the character in the same position in the OPTIMAL string. These values 
      are summed and then returned. 
    """

    def cost(self, city1, city2):
        if (city1 == 1 and city2 == 2) or (city1 == 2 and city2 == 1):
            return 10
        elif (city1 == 1 and city2 == 3) or (city1 == 3 and city2 == 1):
            return 20
        elif (city1 == 1 and city2 == 4) or (city1 == 4 and city2 == 1):
            return 23
        elif (city1 == 1 and city2 == 5) or (city1 == 5 and city2 == 1):
            return 53
        elif (city1 == 1 and city2 == 6) or (city1 == 6 and city2 == 1):
            return 12
        elif (city1 == 2 and city2 == 3) or (city1 == 3 and city2 == 2):
            return 4
        elif (city1 == 2 and city2 == 4) or (city1 == 4 and city2 == 2):
            return 15
        elif (city1 == 2 and city2 == 5) or (city1 == 5 and city2 == 2):
            return 32
        elif (city1 == 2 and city2 == 6) or (city1 == 6 and city2 == 2):
            return 17
        elif (city1 == 3 and city2 == 4) or (city1 == 4 and city2 == 3):
            return 11
        elif (city1 == 3 and city2 == 5) or (city1 == 5 and city2 == 3):
            return 18
        elif (city1 == 3 and city2 == 6) or (city1 == 6 and city2 == 3):
            return 21
        elif (city1 == 4 and city2 == 5) or (city1 == 5 and city2 == 4):
            return 9
        elif (city1 == 4 and city2 == 6) or (city1 == 6 and city2 == 4):
            return 5
        else:
            return 15

    # complete fitness function
    """The fitness function is sum of distance between every two cities in the list. """
    def fitness(self, dna):
       di=0
       for c in range(self.DNA_SIZE-1):
            di += self.cost(dna[c], dna[c+1])
       di+=self.cost(1, dna[self.DNA_SIZE-1])
       return di

    def mutate(self, dna, random1, random2):
        for i in range(self.DNA_SIZE):
            if random1<0.01:
                temp = dna[int(random2)]
                dna[int(random2)]=dna[i]
                dna[i]=temp
        return dna


    def crossover(self, dna1, dna2, random1, random2):
        DNA_SIZE=self.DNA_SIZE
        rr1=int(random1*DNA_SIZE)
        rr2=int(random2*DNA_SIZE)
        
        DNA1_1=dna1[:rr1]
        DNA1_2 =[]
        for i in range(DNA_SIZE):
            if dna2[i]not in DNA1_1:
                DNA1_2.append(dna2[i])
                
        DNA2_1=dna2[:rr2]
        DNA2_2 =[]
        for i in range(DNA_SIZE):
            if dna1[i]not in DNA2_1:
                DNA2_2.append(dna1[i])          
    
        return (DNA1_1+DNA1_2,DNA2_1+DNA2_2)

# endregion
#################################### Algorithms Main Functions #####################################
# region Search_Algorithms_Main_Fn
def SearchAlgorithm_Main():
    searchAlgo = SearchAlgorithms('S,.,.,#,.,.,. .,#,.,.,.,#,. .,#,.,.,.,.,. .,.,#,#,.,.,. #,.,#,E,.,#,.',
                                  [0, 15, 2, 100, 60, 35, 30, 3
                                          , 100, 2, 15, 60, 100, 30, 2
                                          , 100, 2, 2, 2, 40, 30, 2, 2
                                          , 100, 100, 3, 15, 30, 100, 2
                                          , 100, 0, 2, 100, 30])
    fullPath, path, TotalCost = searchAlgo.AstarManhattanHeuristic()
    print('**ASTAR with Manhattan Heuristic ** Full Path:' + str(fullPath) + '\nPath is: ' + str(path)
          + '\nTotal Cost: ' + str(TotalCost) + '\n\n')


# endregion

# region KNN_MAIN_FN
'''The dataset classifies tumors into two categories (malignant and benign) (i.e. malignant = 0 and benign = 1)
    contains something like 30 features.
'''


def KNN_Main():
    BC = load_breast_cancer()
    X = []

    for index, row in pd.DataFrame(BC.data, columns=BC.feature_names).iterrows():
        temp = []
        temp.append(row['mean area'])
        temp.append(row['mean compactness'])
        X.append(temp)
    y = pd.Categorical.from_codes(BC.target, BC.target_names)
    y = pd.get_dummies(y, drop_first=True)
    YTemp = []
    for index, row in y.iterrows():
        YTemp.append(row[1])
    y = YTemp;
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=1024)
    KNN = KNN_Algorithm(7);
    accuracy = KNN.KNN(X_train, X_test, y_train, y_test)
    print("KNN Accuracy: " + str(accuracy))


# endregion

# region Genetic_Algorithm_Main_Fn
def GeneticAlgorithm_Main():
    genetic = GeneticAlgorithm();
    population = genetic.random_population()
    for generation in range(genetic.GENERATIONS):
        # print("Generation %s... Random sample: '%s'" % (generation, population[0]))
        weighted_population = []

        for individual in population:
            fitness_val = genetic.fitness(individual)

            pair = (individual, 1.0 / fitness_val)
            weighted_population.append(pair)
        population = []

        for _ in range(int(genetic.POP_SIZE / 2)):
            ind1 = genetic.weighted_choice(weighted_population)
            ind2 = genetic.weighted_choice(weighted_population)
            ind1, ind2 = genetic.crossover(ind1, ind2, r.random(),r.random())
            population.append(genetic.mutate(ind1,r.random(),r.random()))
            population.append(genetic.mutate(ind2,r.random(),r.random()))

    fittest_string = population[0]
    minimum_fitness = genetic.fitness(population[0])
    for individual in population:
        ind_fitness = genetic.fitness(individual)
    if ind_fitness <= minimum_fitness:
        fittest_string = individual
        minimum_fitness = ind_fitness

    print(fittest_string)
    print(genetic.fitness(fittest_string))


# endregion
######################## MAIN ###########################33
if __name__ == '__main__':

    SearchAlgorithm_Main()
    KNN_Main()
    GeneticAlgorithm_Main()
