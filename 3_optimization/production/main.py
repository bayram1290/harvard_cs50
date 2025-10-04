import scipy.optimize as opt

# Problem description:

# --------------------------------------

# 1)
# Two machines produce x1 and x2.
# x1 costs $50 per hour, and x2 costs $80 per hour.
# Goal is to minimize the total cost.

# 2)
# x1 requires 5 units of labor per hour, and x2 requires 2 units of labor per hour.
# Total of 20 units of labor to spend

# 3)
# x1 requires 10 units of materials per hour, and x2 requires 12 units of materials per hour.
# Total of 90 units of materials to spend

# --------------------------------------

# From:
# 1) Objective function: 50x_1 + 80x_2
# 2) Constraint 1: 5x_1 + 2x_2 <= 20
# 3) Constraint 2: 10x_1 + 12x_2 >= 90 <=> (-10x_1) + (-12x_2) <= -90

# --------------------------------------


result = opt.linprog(
    c = [50, 80], # Cost function: 50x_1 + 80x_2
    A_ub = [[5, 2], [-10, -12]], # Coefficients for inequalities
    b_ub = [20, -90] # Constraints for inequalities: 20 and -90
)

if result.success:
    print(f'X1: {round(result.x[0], 2)} hours') # X1: 1.5 hour
    print(f'X2: {round(result.x[2], 2)} hours') # X2: 6.25 hours
else:
    print('No solution')

# Answer:
# 1) X1: 1.5 hour
# 2) X2: 6.25 hours