import time
from constraint import *

start_process_time = time.perf_counter()

search_problem = Problem()

search_problem.addVariables(
    ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
    ['Monday', 'Tuesday', 'Wednesday']
)

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

for x, y in constraints:
    search_problem.addConstraint(lambda x, y: x != y, (x, y))

for search_solution in search_problem.getSolutions():
    print(search_solution)

end_process_time = time.perf_counter()
print(f"Elapsed time: {end_process_time - start_process_time:.6f} seconds")