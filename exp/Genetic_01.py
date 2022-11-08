# data
import numpy as np
import pandas as pd

restaurants = pd.read_csv("https://raw.githubusercontent.com/xoyeon/mdrplib/master/public_instances/0o100t100s1p100/restaurants.txt", sep = '\t')
orders = pd.read_csv("https://raw.githubusercontent.com/xoyeon/mdrplib/master/public_instances/0o100t100s1p100/orders.txt", sep = '\t')
couriers = pd.read_csv("https://raw.githubusercontent.com/xoyeon/mdrplib/master/public_instances/0o100t100s1p100/couriers.txt", sep = '\t')
rest_orders = pd.merge(restaurants, orders, how='inner', on=['restaurant'])
rest_orders.columns = ['restaurant', 'rx', 'ry', 'order', 'ox', 'oy', 'placement_time', 'ready_time']

restaurants = rest_orders[['rx', 'ry']].to_records(index=False)
orders = rest_orders[['ox', 'oy']].to_records(index=False)
couriers = couriers[['x', 'y']].to_records(index=False)

boundary = 5 # km
speed = 50
costPerKilo = 2000
wage = 9160

# Genetic AL
from random import random, shuffle, randint
from math import floor
import matplotlib.pyplot as plt

# constants
GENERATION_COUNT = 1000 # 한 세대, 가능해들의 집합
POPULATION_COUNT = 500 # 한 generation의 총 인구수
MUTATION_PROBABILITY = 0.9
SATURATION_PERCENTAGE = 0.5

# creating a path object
class Path:
    def __init__(self, sequence):
        self.sequence = sequence
        self.distance = 0 
        self.fitness = 0

    def __repr__(self):
        return "{ " + f"Path: {self.sequence}, Fitness: {self.fitness}" + " }"

# initialization
# Create an initial population.
# This population is usually randomly generated and can be any desired size, from only a few individuals to thousands.
def initialization(path, populationCount):
    population = [path]
    for i in range(populationCount - 1):
        newPath = path.sequence[:]
        while pathExists(newPath, population):
            shuffle(newPath)
        population.append(Path(newPath))
    return population

# Returns true if the path exists and false otherwise
def pathExists(path, population):
    for item in population:
        if item.sequence == path:
            return True
    return False

# evaluation
# Each member of the population is then evaluated and we calculate a 'fitness' for that individual.
# The fitness value is calculated by how well it fits with our desired requirements.
# These requirements could be simple, 'faster algorithms are better', or more complex, 'stronger materials are better but they shouldn't be too heavy'

def calculateDistance(path):
    total = 0
    for i in range(len(path.sequence)):
        if i == len(path.sequence) - 1:
            distance = abs(restaurants[path.sequence[i]][0] - orders[path.sequence[i]][0]) + abs(
            restaurants[path.sequence[i]][1] - orders[path.sequence[i]][1])
            total += distance
        else:
            distance = abs(restaurants[path.sequence[i+1]][0] - orders[path.sequence[i+1]][0]) + abs(
            restaurants[path.sequence[i+1]][1] - orders[path.sequence[i+1]][1])
            total += distance  
    path.distance = total
    return total

def calculateFitness(population):
    fit = distCost = timeCost = 0
    for path in population:
        distance = calculateDistance(path)
        distCost = distance * costPerKilo
        
        timeCost = (distance / speed) * wage
        
        fit += distCost + timeCost
        path.fitness = 1/fit
    for path in population:
        path.fitness /= fit
    return sorted(population, key=lambda x: x.fitness, reverse=True)

# Selection
# We want to be constantly improving our populations overall fitness.
# Selection helps us to do this by discarding the bad designs and only keeping the best individuals in the population.
# There are a few different selection methods but the basic idea is the same, make it more likely that fitter individuals will be selected for our next generation.
def select(population):
    randomNumber = random()
    third = floor(0.3 * len(population))
    randomIndex = randint(0, third)
    if randomNumber <= 0.7:
        return population[randomIndex]
    else:
        return population[randint(third+1, len(population) - 1)]

# Crossover
# During crossover we create new individuals by combining aspects of ourselected individuals.
# from two or more individuals we will create an even 'fitter' offspring which will inherit the best traits from We can think of this as mimicking how sex works in nature.
# The hope is that by combining certain traits each of its parents.
def crossOver(population):   ### --> K-means + Binary 추가하기
    father = select(population)
    mother = select(population)
    # time_success = [] --> 1
    # time_failed = [] --> 0
    while(mother == father):
        mother = select(population)
    startIndex = randint(0,len(mother.sequence) - 2)
    endIndex = randint(startIndex + 1, len(mother.sequence) - 1)
    childSequence = [None] * len(population[0].sequence)
    for i in range(startIndex, endIndex + 1):
        childSequence[i] = mother.sequence[i]
    for i in range(len(childSequence)):
        if childSequence[i] is None:
            for j in range(0, len(childSequence)):
                if father.sequence[j] not in childSequence:
                    childSequence[i] = father.sequence[j]
                    break
    return Path(childSequence)

def crossOverTwoHalfandHalf(population):
    father = select(population)
    mother = select(population)
    while(mother == father):
        mother = select(population)
    mid = len(mother.sequence) // 2
    childSequence = [None] * len(mother.sequence)
    for i in range(mid):
        childSequence[i] = mother.sequence[i]
    for i in range(mid, len(father.sequence)):
        for k in range(len(father.sequence)): 
            if father.sequence[k] not in childSequence:
                childSequence[i] = father.sequence[k]
                break 
    return Path(childSequence)
    
# Mutation 
# We need to add a little bit randomness into our populations' genetics otherwise every combination of solutions we can create would be in our initial population.
# Mutation typically works by making very small changes at random to an individual’s genome.
def mutation(path):
    firstIndex = randint(0, len(path.sequence) - 1)
    secondIndex = randint(0, len(path.sequence) - 1)
    while secondIndex == firstIndex:
        secondIndex = randint(0, len(path.sequence) - 1)
    probability = random()
    if probability < MUTATION_PROBABILITY:
        temp = path.sequence[firstIndex]
        path.sequence[firstIndex] = path.sequence[secondIndex]
        path.sequence[secondIndex] = temp
    return path

def mutationTwoInsertion(path):
    firstIndex = randint(0, len(path.sequence) - 1)
    secondIndex = randint(0, len(path.sequence) - 1)
    while secondIndex == firstIndex:
        secondIndex = randint(0, len(path.sequence) - 1)
    probability = random()
    if probability < MUTATION_PROBABILITY:
        city = path.sequence[firstIndex]
        path.sequence.remove(path.sequence[firstIndex])
        path.sequence.insert(secondIndex, city)
    return path

# Repeat 
# Now we have our next generation we can start again from step two until we reach a termination condition
def geneticAlgorithm(path, populationCount, generationCount):
    path = Path(path)
    population = initialization(path, populationCount)
    population = calculateFitness(population)
    best = population[0]
    print(f"Generation 1: {best.fitness}, distance: {round(best.distance, 2)}")  #, time: {round(best.time, 2)}")
    saturation = 0
    for i in range(2, generationCount + 1):
        print(f"Generation {i}: {best.fitness}, distance: {round(best.distance, 2)}") #, time: {round(best.time, 2)}")
        newGeneration = []
        for _ in range(populationCount):
            child = crossOver(population)
            # child = crossOverTwoHalfandHalf(population)
            newGeneration.append(mutation(child))
            # newGeneration.append(mutationTwoInsertion(child))
        population = calculateFitness(newGeneration)
        if population[0].fitness > best.fitness:
            best = population[0]
            saturation = 0
        else:
            saturation += 1
        if saturation > (SATURATION_PERCENTAGE * GENERATION_COUNT):
            break
    return best


# Plotting the best path found
def plotData(path):  ## restaurants, couriers, orders, 좌표 표시 --> 지도 생성
    x = []
    y = []

    for i in range(len(path.sequence)):
        x.append(restaurants[path.sequence[i]][0])
        y.append(restaurants[path.sequence[i]][1])
    for i in range(len(path.sequence)):
        x.append(orders[path.sequence[i]][0])
        y.append(orders[path.sequence[i]][1])
        
    x.append(restaurants[path.sequence[0]][0])
    x.append(orders[path.sequence[0]][0])
    y.append(restaurants[path.sequence[0]][1])
    y.append(orders[path.sequence[0]][1])

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"The Meal Delivery Courier Routing\nSequence:\n{path.sequence}")
    plt.figure(figsize = (12,12))
    plt.plot(x, y, "bo-")
    plt.show()

# program entry point 
if __name__ == "__main__":        
    cities = list(range(113))
    best = geneticAlgorithm(cities, POPULATION_COUNT, GENERATION_COUNT)
    plotData(best)
