"""
parameter tuning 1
changed:
inputs
corresponding places for inputs
"""
from DCEpy.Features.GardnerStudy.deap import base
from DCEpy.Features.GardnerStudy.deap import creator
from DCEpy.Features.GardnerStudy.deap import tools
from DCEpy.Features.GardnerStudy.deap import algorithms
import random
import numpy as np

def fitness_fun(individual):

    score = sum([np.random.uniform(0.02,0.2), np.random.uniform(0.25,10), np.random.uniform(0.3,1), np.random.uniform(10,100), np.random.uniform(10,200)])

    return score,

# input type mate, select, rates
def parameter_tuning_1(cxtype, selecttype, CXPB, indpb_prob, flagselect, flagcx):

    f_s=1
    # creating types, defining genes, defining inds and pops...

    # creating types
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()

    # defining genes
    # ranges are given by Gardner paper
    t_per_min = 10 * f_s
    t_per_max = 200 * f_s
    MIN = [0.02, 0.25, 0.3, 10, t_per_min]
    MAX = [.2, 10, 1, 100, t_per_max]
    toolbox.register("attr_v", random.uniform, MIN[0], MAX[0])
    toolbox.register("attr_g", random.uniform, MIN[1], MAX[1])
    toolbox.register("attr_p", random.uniform, MIN[2], MAX[2])
    toolbox.register("attr_N", random.uniform, MIN[3], MAX[3])
    toolbox.register("attr_T", random.uniform, MIN[4], MAX[4])

    # defining an individual as a group of the five genes
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_v, toolbox.attr_g, toolbox.attr_p, toolbox.attr_N, toolbox.attr_T), 1)

    # defining the population as a list of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    #toolbox.decorate("population", within_constraints_pi(MIN, MAX))

    # register the crossover operator
    toolbox.register("mate", cxtype)
    # toolbox.decorate("mate", within_constraints_pi(MIN, MAX))

    # register a mutation operator with a probability to mutate of 0.05
    # can change: mu, sigma, and indpb
    # there are others, more particular than this
    mu = np.tile([0],5)
    sigma = np.tile([.06, 3.25, .23, 30, 63.33*f_s],1)
    indpb = np.tile([indpb_prob],5)
    toolbox.register("mutate", tools.mutGaussian, mu.tolist(), sigma.tolist(), indpb.tolist())
    #toolbox.decorate("mutate", within_constraints_pi(MIN,MAX))

    # operator for selecting individuals for breeding the next generation
    if flagselect==0:
        toolbox.register("select", selecttype, tournsize=3)
    else:
        toolbox.register("select", selecttype)

#   #take CXPB as input instead
    MUTPB=0.1
    # NGEN  is the number of generations until final parameters are picked
    NGEN = 40

    # initiate lists to keep track of the best genes(best pi) and best fitness scores for everychannel
    best_genes_all_channels = []
    best_fitness_all_channels = []

    for channel in range(5):

        # create an initial population of size 20
        pop = toolbox.population(n=20)

        # find X_channel (the channel layer in X_train)
        #n = len(X_train)
        #X_channel = np.ones((n, 3, 1))

        #for i in range(n):
        #    for j in range(3):
        #        X_channel[i][j][0] = X_train[i][j][channel]

        # register the fitness function
        toolbox.register("evaluate",lambda x: fitness_fun(x))

        # find the fitness of every individual in the population
        fitnesses = list(map(toolbox.evaluate, pop))

        # assigning each fitness to the individual it represents
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # defining variables to keep track of best indivudals throughout species for a specific channel
        best_species_genes = tools.selBest(pop, 1)[0]
        best_species_value = best_species_genes.fitness.values
        best_gen = 0

        next_mean = 1
        prev_mean = 0

        # start evolution
        for g in range(NGEN):
            print '\t\tWorking on generation %d' % (g + 1)
            if abs(next_mean - prev_mean) > 0.5:

                prev_mean = next_mean
                # Select the next generation's parents
                parents = toolbox.select(pop, k=len(pop))

                # Clone the parents and call them offspring: crossover and mutation will be performed below
                offspring = list(map(toolbox.clone, parents))

                # Apply crossover to children in offspring with probability CXPB
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    # cross two individuals with probability CXPB
                    if random.random() < CXPB:
                        if flagcx==2:
                            toolbox.mate(child1,child2,0.1)
                        else:
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
                # updating best species values
                if max(fitnesses) > best_species_value:
                    best_species_genes = tools.selBest(pop, 1)[0]
                    best_species_value = best_species_genes.fitness.values
                    best_gen = g
                # best_next_obj = max(fitnesses)
                fits = [ind.fitness.values[0] for ind in pop]
                length = len(pop)
                next_mean = sum(fits) / length

        best_genes_all_channels += best_species_genes

        best_fitness_all_channels.append(best_species_value[0])

        T_pers = best_genes_all_channels[4::5]
        max_T = max(T_pers)

        # set all T_per to the max_T
        for i in range(4, len(best_genes_all_channels), 5):
            best_genes_all_channels[i] = max_T

        #weights = weights_tuning(best_genes_all_channels, best_fitness_all_channels, X_train, test_files, win_len,win_overlap, f_s, seizure_time, num_channels=145)[0]
        # print("Best individual in final population is %s with fitness value %s" % (best_ind_finalgen, best_ind_finalgen.fitness.values))
        # print("Channel "+str(channel)+"Best individual in species occurred during generation %s with fitness %s" %(best_gen,best_species_value))

    # will return v for all channels, g for all channels, p for all channels, adapt_rate for all channels, and Tper for all channels
    return best_species_genes[0::5], best_species_genes[1::5], best_species_genes[2::5], best_species_genes[
                                                                                         3::5], best_species_genes[
                                                                                                4::5], sum(best_fitness_all_channels)
# took weights out of output


"""
ideal_probabilities is a helper function that determines the best indpb and CXPB for each method that is looped above.
CXPB range= 0.65-0.95; indpb range=0.01-0.05 from Patil and Pawar Paper --> Grefenstette and Schaffer
"""

def ideal_probabilities(cxtype, selecttype,flagselect, flagcx):
    #initialize storage lists
    best_fitnesses=list()

    #list of CXPB and indpb probabilities
    CXPB=[0.75, 0.80, 0.85, 0.90, 0.95]
    #chris used 0.01 as his indpb, but since MUTPB is 0.1, a parameter of an individual will have 0.1*indpb probability of being mutated.
    indpb=[0.5, 0.4, 0.3, 0.2, 0.1]


    # for 10 iterations
    for i in range(0,5):
        # input in parameter_tuning_1 and obtain the total fitness score of the strongest individual
        [a,b,c,d,e,best_fitness_all_channels]=parameter_tuning_1(cxtype, selecttype, CXPB[i], indpb[i], flagselect, flagcx)
        # store fitnesses from each run
        best_fitnesses.append(best_fitness_all_channels)

    # determine highest fitness score
    best_fitness=max(best_fitnesses)

    # get index of highest fitness score
    best_fitness_index=best_fitnesses.index(best_fitness)

    # use index of highest fitness score to find the corresponding CXPB and indpb that produced the high fitness score.
    best_CXPB=CXPB[best_fitness_index]
    best_indpb=indpb[best_fitness_index]

    # outputs the best fitness score and the corresponding CXPB and indpb that produced it.
    return best_CXPB, best_indpb, best_fitness

"""
parameter_utility tests 6 methods:
cxOnePoint+Tournament, cxOnePoint+Roulette, cxTwoPoint+Tournament, cxTwoPoint+Roulette, cxUniform+Tournament, cxUniform+Roulette,
then delegates to ideal_probabilities to get the best indpb and CXPB and their corresponding fitness for each method.
Finally, identifies the maximum fitness out of the six methods and returns the method and its indpb and CXPB that yielded the best fitness.
"""


def parameter_utility():
    #list of 6 methods: cxOnePoint+Tournament, cxOnePoint+Roulette, cxTwoPoint+Tournament, cxTwoPoint+Roulette, cxUniform+Tournament, cxUniform+Roulette
    methods = ['cxOnePoint+Tournament', 'cxOnePoint+Roulette', 'cxTwoPoint+Tournament', 'cxTwoPoint+Roulette',
               'cxUniform+Tournament', 'cxUniform_Roulette']

    #cx and select method lists for for loop
    cxmethod=[tools.cxOnePoint, tools.cxTwoPoint, tools.cxUniform]
    selectmethod=[tools.selTournament, tools.selRoulette]

    #initialize storage lists
    all_best_CXPB=list()
    all_best_indpb=list()
    all_best_fitness=list()
    """
    all_best_intCXPB=list()
    all_best_intindpb=list()
    all_best_intfitness=list()
    """
    #loop through the three cxmethods
    for j in range(0,3):
        #loop through the two selectmethods
        for i in range(0,2):
            #get max fit individual and their CXPB and indpb probabilities from ideal_probabilities
            [best_CXPB, best_indpb, best_fitness]=ideal_probabilities(cxmethod[j], selectmethod[i], i, j)
            #store best fitness, CXPB, indpb outputted by each selectmethod and cxmethod
            all_best_CXPB.append(best_CXPB)
            all_best_indpb.append(best_indpb)
            all_best_fitness.append(best_fitness)
        """
        all_best_fitness.append(all_best_intfitness)
        all_best_CXPB.append(all_best_intCXPB)
        all_best_indpb.append(all_best_intindpb)
        """
    #get maximum value of best_fitness
    best_best_fitness=max(all_best_fitness)

    #get index of maximum best_fitness
    best_best_fitness_index=all_best_fitness.index(best_best_fitness)

    #get best CXPB and indpb using index of the best fitness
    best_best_CXPB=all_best_CXPB[best_best_fitness_index]
    best_best_indpb=all_best_indpb[best_best_fitness_index]

    #get best method using index of best fitness
    best_method=methods[best_best_fitness_index]

    #returns value of highest fitness, the best method and its best CXPB and indpb
    return best_best_fitness, best_best_CXPB, best_best_indpb, best_method




if __name__ == '__main__':
   [fitness, cxpb, indpb, method] = parameter_utility()
   result=str([fitness, cxpb, indpb, method])

   f=open('Method and CXPB, indpb Rates.txt','w')
   f.write(result)
