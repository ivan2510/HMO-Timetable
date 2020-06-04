import pickle
from load_utils import *

def parse_students(requests, dump_file_name, students_csv_file="instance_4/student.csv"):

    students = defaultdict(dict)

    with open(students_csv_file) as csv_file, open("./instance_4/"+dump_file_name+"_parsed", 'w') as out_csv_file:
        reader = csv.reader(csv_file)
        writer = csv.writer(out_csv_file)
        writer.writerow(['student_id', 'activity_id', 'swap_weight','group_id','new_group_id'])
        next(reader) # skip header

        for row in reader:
            s = Student(*row)

            if (s.student_id, s.activity_id) in requests:
                new_group_id = requests[(s.student_id, s.activity_id)]
                writer.writerow(list(s[:-1]) + [new_group_id])

            else:
                writer.writerow(s)

    return students

name="30min.p"
requests = pickle.load( open( name, "rb" ) )
requests = {(r.student_id, r.activity_id): r.req_group_id for r in requests}
parse_students(requests, name)
