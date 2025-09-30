from pgmpy.inference import VariableElimination
from model import model

inference = VariableElimination(model)

print("\n" + "="*60)
print("PREDICTIONS GIVEN TRAIN IS DELAYED")
print("="*60)

print("\n1. PROBABILITY OF RAIN TYPES GIVEN TRAIN IS DELAYED")
rain_result = inference.query(variables=['rain'], evidence={'train': 'delayed'})
for state, prob in zip(rain_result.state_names['rain'], rain_result.values):
    print(f"   P(rain = {state:5} | train=delayed) = {prob:.4f}")

print("\n2. PROBABILITY OF MAINTENANCE GIVEN TRAIN IS DELAYED")
maintenance_result = inference.query(variables=['maintenance'], evidence={'train': 'delayed'})
for state, prob in zip(maintenance_result.state_names['maintenance'], maintenance_result.values):
    print(f"   P(maintenance = {state:3} | train=delayed) = {prob:.4f}")

print("\n3. PROBABILITY OF APPOINTMENT OUTCOME GIVEN TRAIN IS DELAYED")
appointment_result = inference.query(variables=['appointment'], evidence={'train': 'delayed'})
for state, prob in zip(appointment_result.state_names['appointment'], appointment_result.values):
    print(f"   P(appointment = {state:5} | train=delayed) = {prob:.4f}")

print("\n4. JOINT PROBABILITY OF RAIN AND MAINTENANCE GIVEN TRAIN IS DELAYED")
joint_result = inference.query(variables=['rain', 'maintenance'], evidence={'train': 'delayed'})
print("   Rain     | Maintenance | Probability")
print("   " + "-" * 35)
for i, rain_state in enumerate(joint_result.state_names['rain']):
    for j, maint_state in enumerate(joint_result.state_names['maintenance']):
        prob = joint_result.values[i, j]
        print(f"   {rain_state:5}    | {maint_state:3}        | {prob:.4f}")

print("\n5. MOST LIKELY SCENARIO GIVEN TRAIN IS DELAYED")
full_joint_result = inference.query(variables=['rain', 'maintenance', 'appointment'], 
                                   evidence={'train': 'delayed'})

max_prob = 0
max_scenario = None

for i, rain_state in enumerate(full_joint_result.state_names['rain']):
    for j, maint_state in enumerate(full_joint_result.state_names['maintenance']):
        for k, appoint_state in enumerate(full_joint_result.state_names['appointment']):
            prob = full_joint_result.values[i, j, k]
            if prob > max_prob:
                max_prob = prob
                max_scenario = (rain_state, maint_state, appoint_state)

print(f"   Most likely scenario: Rain={max_scenario[0]}, Maintenance={max_scenario[1]}, Appointment={max_scenario[2]}")
print(f"   Probability: {max_prob:.4f}")

print("\n6. COMPARISON WITH PRIOR PROBABILITIES (NO EVIDENCE)")
print("   Variable    | State     | Prior     | Given Delayed | Change")
print("   " + "=" * 60)

# Prior probabilities
rain_prior = inference.query(variables=['rain'])
maintenance_prior = inference.query(variables=['maintenance'])
appointment_prior = inference.query(variables=['appointment'])

for i, state in enumerate(rain_prior.state_names['rain']):
    prior = rain_prior.values[i]
    posterior = rain_result.values[i]
    change = posterior - prior
    print(f"   {'rain':11} | {state:6}    | {prior:.4f}    | {posterior:.4f}      | {change:+.4f}")
print("   " + "-" * 60)
for i, state in enumerate(maintenance_prior.state_names['maintenance']):
    prior = maintenance_prior.values[i]
    posterior = maintenance_result.values[i]
    change = posterior - prior
    print(f"   {'maintenance':11} | {state:4}      | {prior:.4f}    | {posterior:.4f}      | {change:+.4f}")
print("   " + "-" * 60)
for i, state in enumerate(appointment_prior.state_names['appointment']):
    prior = appointment_prior.values[i]
    posterior = appointment_result.values[i]
    change = posterior - prior
    print(f"   {'appointment':11} | {state:6}    | {prior:.4f}    | {posterior:.4f}      | {change:+.4f}")

print("\n" + "*"*60)
print("KEY INSIGHTS:")
print("*"*60)
print("• Given train is delayed, probability of heavy rain increases")
print("• Probability of no maintenance increases (maintenance helps prevent delays)")
print("• Probability of missing appointment increases")
print("• The most likely scenario involves: No rain, exists maintenance and attend appointment")