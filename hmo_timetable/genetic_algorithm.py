import random
import array
import time
import numpy as np
import copy
from collections import Counter, deque
from deap import algorithms, base, creator, tools
import pickle


class GeneticAlgorithm:

    def __init__(self,
        eval_function,
        requests,
        num_of_queries=100,
        timeout=600):

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", array.array, typecode="d",
                       fitness=creator.FitnessMax, strategy=None)

        creator.create("Strategy", array.array, typecode="d")
        self.timeout = timeout
        self.requests = requests

        global counter
        counter = 0

        def greedy_mutate(individual):
            if (random.random() < 0.1):
                for i in range(len(individual)):
                    if random.random() < 1/num_of_queries:
                        individual[i] = type(individual[i])(not individual[i])

            else:
                interval = 1
                individual = greedy(ind_array=individual, timeout_section=timeout*(1/interval), fit_goal=0.0, verbose=False)

            return individual,

        def greedy(ind_array=np.zeros(num_of_queries, dtype=int), timeout_section=4, fit_goal=-8000, return_fit=False, verbose=True):
            start_time = time.time()
            tabu = set()
            fails = 0
            best_fit = -10666
            start_pos = 0

            while fails < 100:

                if ind_array[start_pos]!=1:
                    ind_array[start_pos] = 1

                    fit = eval_function(ind_array)[0]
                    valid = fit > -99999
                    better = fit > best_fit and valid
                    if not better:
                        ind_array[start_pos] = 0
                        fails+=1
                    else:
                        best_fit = fit
                        fails=0

                start_pos = np.random.randint(num_of_queries)

                while start_pos in tabu:
                    start_pos = np.random.randint(num_of_queries)

                tabu.add(start_pos)


                if len(tabu) >= 500:
                    tabu = set()

                if verbose:
                    print("tabu_size {} time {} fittnes {}".format(len(tabu), time.time()-start_time, eval_function(ind_array)[0]))

                if (time.time()-start_time) > (timeout / timeout_section):
                    break

                if best_fit > fit_goal:
                    break

            if return_fit:
                return ind_array, best_fit
            else:
                return ind_array

        print('greed init')
        greedy_init, best_fit = greedy( fit_goal=0, timeout_section=timeout*(1/(3*60)), return_fit=True , verbose=True)
        print("init fit:", best_fit)

        def initES(icls, scls):
            print('g create pop')
            ind_array, fit = greedy(ind_array=greedy_init, timeout_section=timeout*(1/30), verbose=False, return_fit=True)
            ind = icls(ind_array)
            ind.strategy = ind.strategy = scls(random.uniform(0.1, 0.15) for _ in range(num_of_queries))

            print("init fit:", fit)

            return ind

        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", initES, creator.Individual, creator.Strategy)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", eval_function)

        # self.toolbox.register("mate", tools.cxOnePoint)
        # self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mate", switch)
        self.hof = tools.HallOfFame(1, similar=np.array_equal)

        # self.toolbox.register("mutate", tools.mutFlipBit, indpb=1/num_of_queries)
        self.toolbox.register("mutate", greedy_mutate)
        # self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("select", tools.selAutomaticEpsilonLexicase)
        # self.toolbox.register("select", tools.selBest)


    def run(self, start_time_main, population=3, ngen=10000, cxpb=0.8, mutpb=0.2):
        total_start_time = time.time()

        pop = self.toolbox.population(n=population)

        #print("Init time: " + str(init_time))

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("max", np.max)

        # algorithms.eaSimple(pop, self.toolbox, cxpb=cxpb, mutpb=mutpb, ngen=500, stats=stats,
        #                     halloffame=self.hof)

        eaMuPlusLambda(
            pop,
            self.toolbox,
            mu=3,
            lambda_=10,
            cxpb=cxpb,
            mutpb=mutpb,
            ngen=100000,
            stats=stats,
            halloffame=self.hof,
            verbose=True,
            start_time_main=start_time_main,
            requests=self.requests
        )

        # algorithms.eaMuCommaLambda(
        #     pop,
        #     self.toolbox,
        #     mu=1,
        #     lambda_=10,
        #     cxpb=cxpb,
        #     mutpb=mutpb,
        #     ngen=1000,
        #     stats=stats,
        #     halloffame=self.hof
        # )
        # print("total_time:", time.time()-total_start_time)

        return tools.selBest(pop, k=1)

def switch(ind1, ind2, swaps=10):
    indices = np.random.randint(len(ind1), size=swaps)
    for i in indices:
        ind1[i] = ind2[i]

    indices = np.random.randint(len(ind1), size=swaps)
    for i in indices:
        ind2[i] = ind1[i]

    return ind1, ind2


def calculate_fitness(swap_weight_for_requests, number_of_requests_per_student, count_number_of_swaps_per_st_id, cnts_n, min_p, max_p, minmax_penalty, award_student, award_activity):

    global counter

    counter += 1

    awards = 0
    for key in count_number_of_swaps_per_st_id.keys():
        student_requests = number_of_requests_per_student[key]
        requests_done = count_number_of_swaps_per_st_id[key]
        if(len(award_activity) < requests_done):
            awards = awards + award_activity[-1]
        else:
            awards = awards + award_activity[requests_done - 1]
        awards = awards + int(student_requests==requests_done)*award_student

    penalty = minmax_penalty*(
        np.sum(np.maximum(min_p - cnts_n, 0)) + np.sum(np.maximum(cnts_n - max_p, 0))
    )

    result = sum(swap_weight_for_requests) + awards - penalty
    #print("Result: " + str(result))
    return (result, )


def check_valid(individual, limits, requests, students, overlaps, cnts, mins, maxs, groups_id_to_idx):
    # limits
    cnts_n = copy.deepcopy(cnts)  # copying limits because we will modify the list

    indices = np.where(individual)[0]

    selected_requests = [requests[i] for i in indices]

    swap_weight_for_requests = [int(students[request.student_id][request.activity_id].swap_weight) for request in selected_requests]

    students_ids = [request.student_id for request in selected_requests] #get the student_ids to calculate number of requests done for student

    count_number_of_swaps_per_st_id = Counter(students_ids)

    inc_groups_ids = [request.req_group_id for request in selected_requests]
    inc_groups_idxs = [groups_id_to_idx[id] for id in inc_groups_ids if id in groups_id_to_idx]  # todo: popravi kako ovo rjesiti


    dec_groups_ids = [
        students[request.student_id][request.activity_id].group_id for request in selected_requests
        if request.student_id in students and request.activity_id in students[request.student_id]
    ]

    assert len(inc_groups_ids) == len(dec_groups_ids)

    dec_groups_idxs = [groups_id_to_idx[id] for id in dec_groups_ids if id in groups_id_to_idx]

    for idx in inc_groups_idxs:
        cnts_n[idx]+=1
    for idx in dec_groups_idxs:
        cnts_n[idx]-=1

    invalid_limits = np.sum(cnts_n > maxs) + np.sum(cnts_n < mins)


    # overlaps
    transfered_students_ids = [request.student_id for request in selected_requests]
    activities_ids = [request.activity_id for request in selected_requests]

    invalid_accepted_requests_count = 0
    for old_group_id, activity_id, student_id, new_group_id in zip(dec_groups_ids, activities_ids, transfered_students_ids, inc_groups_ids):
        activities = list(students[student_id].keys())
        overlaps_group = []
        for act in activities:
            if act != activity_id:
                overlaps_group = overlaps_group + overlaps[students[student_id][act].group_id]

        if new_group_id in overlaps_group:
            invalid_accepted_requests_count += 1

    invalid_overlaps = invalid_accepted_requests_count

    stud_act = list(zip(transfered_students_ids, activities_ids))

    invalid_doubles = len(stud_act)-len(set(stud_act))

    # print("overlaps:", time.time()-start_ol)

    invalid_count = invalid_limits + invalid_overlaps + invalid_doubles

    valid = invalid_count <= 0

    # print("total_time:", time.time()-start)

    return valid, invalid_count, swap_weight_for_requests, count_number_of_swaps_per_st_id, cnts_n

def dump(file_name, best_instance, requests):
    global counter
    best_instance = best_instance[0]
    filtered_requests = [r for r, b in zip(requests, best_instance) if b == 1]
    pickle.dump(filtered_requests, open(file_name+".p", "wb"))

    with open(file_name+'_num_evals.txt',"w") as f:
        f.write(str(counter))




def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                   stats=None, halloffame=None, verbose=__debug__, start_time_main=None, requests=None):

    dumps = [False,False,False]
    start_time = time.time()
    best = -1e5

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)
    # Begin the generational process
    for gen in range(1, ngen + 1):
        gen_start = time.time()
        # Vary the population

        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)
        population[-1]=offspring[0]

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        print("\n")
        print("gen time:", time.time()-gen_start)
        print("Tot time:", time.time()-start_time)
        total_run_time = time.time() - start_time_main
        print("App total run time:", total_run_time)

        if total_run_time > 9*60 and not dumps[0]:
            dump("10min", offspring, requests)
            dumps[0]=True

        if total_run_time > 29*60 and not dumps[1]:
            dump("30min", offspring, requests)
            dumps[1]=True

        if total_run_time > 59*60 and not dumps[2]:
            dump("60min", offspring, requests)
            dumps[2]=True
            return population, logbook


    return population, logbook