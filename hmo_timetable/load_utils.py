import csv
from IPython import embed

from collections import namedtuple, defaultdict, Counter
import numpy as np

Limit = namedtuple('Limit', 'group_id students_cnt min min_preferred max max_preferred')
Request = namedtuple('Request', 'student_id activity_id req_group_id curr_group_id')
Student = namedtuple('Student', 'student_id activity_id swap_weight group_id new_group_id')

def load_limits(limits_csv_file="instance_2/limits.csv"):

    limits = dict()
    with open(limits_csv_file) as csv_file:
        reader = csv.reader(csv_file)
        next(reader) # skip header
        for row in reader:
            limits[row[0]]=Limit(*row)

    group_id_idx_mappings = {id: int(i) for i, id in enumerate(limits.keys())}
    group_idx_id_mappings = {int(i): id for i, id in enumerate(limits.keys())}

    # counts = [limits[group_idx_id_mappings[i]].students_cnt for i in range(len(limits.keys()))]
    cnts, mins, maxs, min_p, max_p = map(
        np.array,
        zip(*[
            (
                int(limits[id].students_cnt), int(limits[id].min), int(limits[id].max), int(limits[id].min_preferred), int(limits[id].max_preferred)
            ) for id in group_idx_id_mappings.values()
        ])
    )

    return limits, group_id_idx_mappings, group_idx_id_mappings, cnts, mins, maxs, min_p, max_p


def load_requests(students, requests_csv_file="instance_2/requests.csv"):

    with open(requests_csv_file) as csv_file:
        reader = csv.reader(csv_file)
        next(reader) # skip header
        requests = [Request(*row, students[row[0]][row[1]].group_id) for row in reader]
        number_of_requests_for_students = set([(row[0], row[1]) for row in reader])
        number_of_requests_per_student = Counter([req[0] for req in number_of_requests_for_students])

    return requests, number_of_requests_per_student


def load_students(students_csv_file="instance_2/student.csv"):

    students = defaultdict(dict)
    with open(students_csv_file) as csv_file:
        reader = csv.reader(csv_file)
        next(reader) # skip header
        for row in reader:
            students[row[0]][row[1]]=Student(*row)

    return students


def load_overlaps(overlaps_csv_file="instance_2/overlaps.csv"):

    overlaps = defaultdict(list)
    with open(overlaps_csv_file) as csv_file:
        reader = csv.reader(csv_file)
        next(reader) # skip header
        for row in reader:
            overlaps[row[0]].append(row[1])

    return overlaps

def preprocess(limits, requests, students):
    index = 0
    indices_to_remove = []
    request_pre = []
    for req in requests:
        demand = 0
        requested_group = req.req_group_id
        current_group = students[req.student_id][req.activity_id].group_id
        demand_max = len([requ for requ in requests if requ.curr_group_id == requested_group])
        demand_min = len([requ for requ in requests if requ.req_group_id == current_group])
        if (demand_min==0 and limits[current_group].students_cnt <= limits[current_group].min) or (demand_max==0 and limits[requested_group].students_cnt >= limits[requested_group].max):
            indices_to_remove.append(index)
        index+=1

    for i in range(len(requests)):
        if i not in indices_to_remove:
            request_pre.append(requests[i])

    return request_pre
