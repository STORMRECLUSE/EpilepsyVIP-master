__author__ = "Sarah"

import random

import numpy as np
from deap import base
from deap import creator
from deap import tools


def within_constraints(MIN, MAX):
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in xrange(len(child)):
                    if MIN[i] > child[i]:
                        child[i] = MIN[i]
                    if MAX[i] < child[i]:
                        child[i] = MAX[i]
            return offspring
        return wrapper
    return decorator


def fitness_fn(individual, X_train, feat_vec, seizure_times, f_s, window_length, window_overlap):

    nu = individual[0]
    gamma = individual[1]
    C = individual[2]
    adapt_rate = individual[3]
    T_per = individual[4]

    S, EDF, FPR,  mu = ef(nu, gamma, C, adapt_rate, T_per, X_train, feat_vec, seizure_times, f_s, window_length, window_overlap)

    alpha1 = 100*S-10*(1-np.sign(S-0.75))
    alpha2 = 20*EDF
    alpha3 = -10*FPR-20*(1-np.sign(5-FPR))
    alpha4 = max(-mu,30)
    result=alpha1+alpha2+alpha3+alpha4

    return result,


def parameter_tuning(X_train, feat_vec, seizure_times, f_s, window_length, window_overlap):

        #creating types
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        t_per_min = 10*f_s
        t_per_max = 200*f_s

        #defining genes
        #ranges are given by Gardner paper
        toolbox.register("attr_v", random.uniform, .02, .2)
        toolbox.register("attr_g", random.uniform, .25, 10)
        toolbox.register("attr_p", random.uniform, .3, 1)
        toolbox.register("attr_N", random.uniform, 10, 100)
        toolbox.register("attr_T", random.uniform, t_per_min, t_per_max)

        #defining an individual as a group of the five genes
        toolbox.register("individual", tools.initCycle, creator.Individual,
                         (toolbox.attr_v, toolbox.attr_g, toolbox.attr_p, toolbox.attr_N, toolbox.attr_T), 1)

        #defining the population as a list of individuals
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # register the fitness function
        toolbox.register("evaluate", fitness_fn(X_train, feat_vec, seizure_times, f_s, window_length, window_overlap))

        MIN = [0.02, 0.25, 0.3, 10, t_per_min]
        MAX = [.2, 10, 1, 100, t_per_max]
        # register the crossover operator
        # other options are: cxOnePoint, cxUniform (requires an indpb input, probably can just use CXPB)
        # there are others, more particular than these options
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.decorate("mate", within_constraints(MIN,MAX))

        # register a mutation operator with a probability to mutate of 0.05
        # can change: mu, sigma, and indpb
        # there are others, more particular than this
        toolbox.register("mutate", tools.mutGaussian, mu=1, sigma=10, indpb=0.03)
        toolbox.decorate("mutate", within_constraints(MIN,MAX))

        # operator for selecting individuals for breeding the next generation
        # other options are: tournament: randonly picks tournsize out of population, chosses fittest, and has that be
        # a parent. continues until number of parents is equal to size of population.
        # there are others, more particular than this
        toolbox.register("select", tools.selTournament, tournsize=3)
        #toolbox.register("select", tools.selRoulette)

        #create an initial population of size 20
        pop = toolbox.population(n=20)

        # CXPB  is the probability with which two individuals are crossed
        CXPB = 0.3

        # MUTPB is the probability for mutating an individual
        MUTPB = 0.5

        # NGEN  is the number of generations until final parameters are picked
        NGEN = 40

        print("Start of evolution")

        # find the fitness of every individual in the population
        fitnesses = list(map(toolbox.evaluate, pop))

        #assigning each fitness to the individual it represents
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        #defining variables to keep track of best indivudals throughout species
        best_species_genes = tools.selBest(pop, 1)[0]
        best_species_value = best_species_genes.fitness.values
        best_gen = 0

        next_mean=1
        prev_mean=0

        #start evolution
        for g in range(NGEN):
            if abs( next_mean - prev_mean ) > 0.005 :

                prev_mean=next_mean
                # Select the next generation's parents
                parents = toolbox.select(pop, len(pop))

                # Clone the parents and call them offspring: crossover and mutation will be performed below
                offspring = list(map(toolbox.clone, parents))

                # Apply crossover to children in offspring with probability CXPB
                for child1, child2 in zip(offspring[::2], offspring[1::2]):

                    # cross two individuals with probability CXPB
                    if random.random() < CXPB:
                        toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values

                # Apply mutation to children in offspring with probability MUTPB
                for mutant in offspring:
                    if random.random() < MUTPB:
                        toolbox.mutate(mutant)
                        del mutant.fitness.values

                # Find the fitnessess for all the children for whom fitness changed
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # Offspring becomes the new population
                pop[:] = offspring

                #updating best species values
                if max(fitnesses):
                    if max(fitnesses) > best_species_value:
                        best_species_genes = tools.selBest(pop, 1)[0]
                        best_species_value = best_species_genes.fitness.values
                        best_gen = g
                    best_next_obj = max(fitnesses)

                fits = [ind.fitness.values[0] for ind in pop]
                length = len(pop)
                next_mean = sum(fits) / length


        best_ind = tools.selBest(pop, 1)[0]
        print("Best individual in final population is %s with fitness value %s" % (best_ind, best_ind.fitness.values))
        print("Best individual in species is %s and occurred during generation %s with fitness %s" %(best_species_genes,best_gen,best_species_value))
        return best_species_genes[0], best_species_genes[1], best_species_genes[2], best_species_genes[3], best_species_genes[4]




# def parameter_tuning_2():
#         creator.create("FitnessMax", base.Fitness, weights=(1.0,))
#         creator.create("Individual", list, fitness=creator.FitnessMax)
#         toolbox = base.Toolbox()
#         toolbox.register("attr_v", random.uniform, .02, .2)
#         toolbox.register("attr_g", random.uniform, .25, 10)
#         toolbox.register("attr_p", random.uniform, .3, 1)
#         toolbox.register("attr_N", random.uniform, 10, 100)
#         toolbox.register("attr_T", random.uniform, 10, 200)
#         toolbox.register("individual", tools.initCycle, creator.Individual,
#                          (toolbox.attr_v, toolbox.attr_g, toolbox.attr_p, toolbox.attr_N, toolbox.attr_T), 1)
#         toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#         toolbox.register("evaluate", fitness_fn)
#         toolbox.register("mate", tools.cxTwoPoint)
#         toolbox.register("mutate", tools.mutGaussian, mu=1, sigma=10, indpb=0.03)
#         toolbox.register("select", tools.selRoulette)
#         pop = toolbox.population(n=20)
#         CXPB = 0.3
#         MUTPB = 0.3
#         NGEN = 40
#         pop = toolbox.population(n=20)
#         hof = tools.HallOfFame(1)
#         stats = tools.Statistics(lambda ind: ind.fitness.values)
#         stats.register("avg", np.mean)
#         stats.register("std", np.std)
#         stats.register("min", np.min)
#         stats.register("max", np.max)
#
#         pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40,
#                                     stats=stats, halloffame=hof, verbose=True)
#         print("%s %s" %(pop, hof))
#         return pop, log, hof

#pop,log,hof = parameter_tuning_2()
