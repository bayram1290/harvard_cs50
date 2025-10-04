import time

def assign_unassigned_variable(assignment):
    for variable in variables:
        if variable not in assignment:
            return variable

def check_consistent(new_assignment):
    for (x, y) in constraints:
        if x not in new_assignment or y not in new_assignment:
            continue

        if new_assignment[x] == new_assignment[y]:
            return False

    return True

def backtrack(assignment):
    if len(assignment) == len(variables):
        return assignment

    var = assign_unassigned_variable(assignment)
    for value in domain:
        new_assignment = assignment.copy()
        new_assignment[var] = value

        if check_consistent(new_assignment):
            result = backtrack(new_assignment)

            if result is not None:
                return result

    return None

start_process_time = time.perf_counter()

variables = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

constraints = [
    ('A', 'B'),
    ('A', 'C'),
    ('B', 'C'),
    ('B', 'D'),
    ('B', 'E'),
    ('C', 'E'),
    ('C', 'F'),
    ('D', 'E'),
    ('E', 'F'),
    ('E', 'G'),
    ('F', 'G'),
]

domain = ['Monday', 'Tuesday', 'Wednesday']


solution = backtrack(dict())
print(solution)
end_process_time = time.perf_counter()
print(f"Elapsed time: {end_process_time - start_process_time:.6f} seconds")