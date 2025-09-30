from model import model
from pgmpy.inference import VariableElimination
from pgmpy.metrics import log_likelihood_score
import numpy as np
inference = VariableElimination(model)

print("\n" + "="*60)
print("LIKELIHOOD PROBABILITY CALCULATION")
print("="*60)
print("Observation:")
print("  - rain: none")
print("  - maintenance: no") 
print("  - train: on time")
print("  - appointment: attend")
print()

print("METHOD 1: Direct Calculation using Chain Rule")
print("-" * 40)



p_rain = inference.query(variables=['rain']).values[0]
print(f"P(rain=none) = {p_rain:.4f}")

p_maintenance = inference.query(variables=['maintenance'], evidence={'rain': 'none'}).values[1]
print(f"P(maintenance=no | rain=none) = {p_maintenance:.4f}")

p_train = inference.query(variables=['train'], evidence={'rain': 'none', 'maintenance': 'no'}).values[0]
print(f"P(train=ontime | rain=none, maintenance=no) = {p_train:.4f}")

p_appointment = inference.query(variables=['appointment'], evidence={'rain': 'none', 'maintenance': 'no', 'train': 'on time'}).values[0]
print(f"P(appointment=attend | train=ontime) = {p_appointment:.4f}")

joint_prob = p_rain * p_maintenance * p_train * p_appointment
print(f"\nJoint probability (chain rule): {joint_prob:.6f}")
print(f"Likelihood P(rain=none, maintenance=no, train=ontime, appointment=attend) = {joint_prob:.6f}")


print("\n" + "="*60)
print("METHOD 2: Using pgmpy's Probability Calculation")
print("-" * 40)
try:
    probability = model.get_state_probability(
        {
            'rain': 'none',
            'maintenance': 'no',
            'train': 'on time',
            'appointment': 'attend'
        }
    )
    print(f"Likelihood probability: {probability:.6f}")
except Exception as e:
    print(f"Error with model.probability(): {e}")
    print("Using alternative approach...")

    # Alternative: Use log probability and convert back
    log_probability = log_likelihood_score(model, [{
        'rain': 'none', 'maintenance': 'no', 'train': 'on time', 'appointment': 'attend'
    }])
    probability = np.exp(log_probability)
    print(f"Likelihood probability (from log): {probability:.6f}")



print("\n" + "="*60)
print("3: Compare with Other Scenarios")
print("-" * 40)

# Compare with other possible observations
scenarios = [
    {'rain': 'none', 'maintenance': 'no', 'train': 'on time', 'appointment': 'attend'},
    {'rain': 'none', 'maintenance': 'no', 'train': 'on time', 'appointment': 'miss'},
    {'rain': 'none', 'maintenance': 'no', 'train': 'delayed', 'appointment': 'attend'},
    {'rain': 'heavy', 'maintenance': 'yes', 'train': 'delayed', 'appointment': 'miss'}
]

print("Probability comparison for different scenarios:")
for i, scenario in enumerate(scenarios, 1):
    try:
        prob = model.get_state_probability(scenario)
        print(f"Scenario {i}: {scenario}")
        # print(f"  Probability: {prob:.6f}")
    except:
        log_prob = log_likelihood_score(model, [scenario])
        prob = np.exp(log_prob)
        print(f"Scenario {i}: {scenario}")
        print(f"  Probability: {prob:.6f}")

print("\n" + "="*60)
print("FINAL RESULT")
print("="*60)
print(f"Likelihood probability P(rain=none, maintenance=no, train=on time, appointment=attend)")
print(f"= {joint_prob:.6f}")
print(f"= {joint_prob * 100:.2f}%")