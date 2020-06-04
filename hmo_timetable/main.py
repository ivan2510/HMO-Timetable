from load_utils import *
from genetic_algorithm import *
import numpy as np
import argparse


def main():
    start_time_main = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-timeout', type=int, default=600, help='time for execution of program')
    parser.add_argument('-award-activity', type=str, default='1,2,4', help='awards for done requests')
    parser.add_argument('-award_student', type=int, default=1, help='award for making all requests of particular student')
    parser.add_argument('-minmax-penalty', type=int, default=1, help='penalty for breaking soft constraints')
    parser.add_argument('-students-file', type=str, default='instance_2/student.csv', help='path to the students file')
    parser.add_argument('-requests-file', type=str, default='instance_2/requests.csv', help='path to the requests file')
    parser.add_argument('-overlaps-file', type=str, default='instance_2/overlaps.csv', help='path to the overlaps file')
    parser.add_argument('-limits-file', type=str, default='instance_2/limits.csv', help='path to the limits file')
    opt = parser.parse_args()

    number_of_evals = 0
    limits, group_id_idx_mappings, group_idx_id_mappings, cnts, mins, maxs, min_p, max_p = load_limits(opt.limits_file)

    # requests = requests[:200]

    students = load_students(opt.students_file)
    overlaps = load_overlaps(opt.overlaps_file)
    requests, number_of_requests_per_student = load_requests(students, opt.requests_file)
    print("number of requests:", len(requests))
    minmax_penalty = opt.minmax_penalty
    award_student = opt.award_student
    award_activity = [int(act) for act in opt.award_activity.rsplit(',')]
    preprocesed_req = preprocess(limits, requests, students)
    print("New number of requests: " + str(len(preprocesed_req)))

    initial_penalty = 0
    for i in range(len(cnts)):
        if(cnts[i] < min_p[i] and cnts[i] >= mins[i]):
            initial_penalty = initial_penalty + (min_p[i] - cnts[i])*minmax_penalty
        if cnts[i] > max_p[i] and cnts[i] <= maxs[i]:
            initial_penalty = initial_penalty + (cnts[i] - max_p[i])*minmax_penalty

    print("Initial penalty: " + str((-1)*initial_penalty))

    def evaluate(individual):
        valid, invalid_count, swap_weight_for_requests, count_number_of_swaps_per_st_id, cnts_n = check_valid(
            individual,
            limits,
            preprocesed_req,
            students,
            overlaps,
            cnts,
            mins,
            maxs,
            group_id_idx_mappings
        )
        if not valid:
            fitness =  (-100000*invalid_count,)

        else:
            fitness =  calculate_fitness(swap_weight_for_requests,
                                        number_of_requests_per_student,
                                        count_number_of_swaps_per_st_id,
                                        cnts_n,
                                        min_p,
                                        max_p,
                                        minmax_penalty,
                                        award_student,
                                        award_activity)

        return fitness

    ga = GeneticAlgorithm(
        evaluate,
        requests=preprocesed_req,
        num_of_queries=len(preprocesed_req),
        timeout=opt.timeout,
    )

    best = ga.run(start_time_main)
    print("best:", np.sum(best))
    print("TOTAL TIME:", time.time()-start_time_main)


if __name__ == "__main__":
    main()
