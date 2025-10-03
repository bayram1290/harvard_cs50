from pgmpy.sampling import BayesianModelSampling
from collections import Counter
from model import model

sampler = BayesianModelSampling(model)

print("=" * 60)
print("REJECTION SAMPLING: Appointment given Train=delayed")
print("=" * 60)

def rejection_sampling(turns=10000):
    data = []
    counter = 0
    while len(data) < turns:
        sample = sampler.forward_sample(size=1, show_progress=False)
        if sample['train'].values[0] == 'delayed':
            data.append(sample['appointment'].values[0])
        counter += 1
    rate = turns / counter
    return data, rate

N = 10000
data, acceptance_rate = rejection_sampling(N)
rejection_dist = Counter(data)

print(f"Generated {N} samples with rejection sampling")
print(f"Total samples needed: {int(N/acceptance_rate):,}")
print(f"Acceptance rate: {acceptance_rate:.4f}")
print(f"Distribution of Appointment, given Train = delayed")
print('Attended: ' + str(rejection_dist['attend']) + ' | Missed: ' + str(rejection_dist['miss']))

print("\n" + "=" * 60)
print("FORWARD SAMPLING WITH FILTERING")
print("=" * 60)
N1 = 50000
forward_samples = sampler.forward_sample(size=N1, show_progress=False)
filtered_samples = [sample for sample in forward_samples.values if sample[2] == 'delayed']
filtered_dist = None

if len(filtered_samples) >= N:
    filtered_data = [sample[3] for sample in filtered_samples[:N]]
    filtered_dist = Counter(filtered_data).most_common(2)
    print(f"Generated {N} filtered samples from {len(forward_samples)} total samples")
    print(f"Attended: {filtered_dist[0][1]} | Missed: {filtered_dist[1][1]}")
else:
    print(f"Not enough samples meeting condition. Got {len(filtered_samples)} out of {len(forward_samples.values)}")


print("\n" + "=" * 60)
print("LIKELIHOOD WEIGHTED SAMPLING")
print("=" * 60)

lw_samples = sampler.likelihood_weighted_sample(evidence=[('train', 'delayed')], size=N, show_progress=False)
lw_data = [sample[3] for sample in lw_samples.itertuples(index=False)]
lw_dist = Counter(lw_data).most_common(2)
print(f"Generated {N} samples using likehood weighted sampling")
print(f"Distribution of Appointment, given Train = delayed")
print(f"Attended: {lw_dist[0][1]} | Missed: {lw_dist[1][1]}")

print("\n" + "=" * 60)
print("PROBABILITY ESTIMATES")
print("=" * 60)

rejection_probs = {k: v/N for k, v in rejection_dist.items()}
lw_probs = {k: v/N for k, v  in Counter(lw_data).items()}

print("Rejection Sampling Probabilities:")
for outcome, prob in rejection_probs.items():
    print(f"    P(appointment={outcome} | train=delayed) = {prob:.4f}")

print("\nLikelihood Weighted Sampling Probabilities:")
for outcome, prob in lw_probs.items():
    print(f"    P(appointment={outcome} | train=delayed) = {prob:.4f}")

print("\n" + "=" * 60)
print("COMPARISON WITH EXACT INFERENCE")
print("=" * 60)

from pgmpy.inference import VariableElimination
inference = VariableElimination(model)
exact_result = inference.query(variables=['appointment'], evidence={'train':'delayed'})

print("Exact probabilities from Variable Elimination:")
for state, prob in zip(exact_result.state_names['appointment'], exact_result.values):
    print(f"    P(appointment={state} | train=delayed) = {prob:.4f}")

print("\nDifferences from exact probabilities:")
for state, exact_prob in zip(exact_result.state_names['appointment'], exact_result.values):
    rejection_difference = abs(rejection_probs.get(state, 0) - exact_prob)
    lw_difference = abs(lw_probs.get(state, 0) - exact_prob)
    print(f"     {state}:")
    print(f"     Rejection sampling error: {rejection_difference:.4f}")
    print(f"     Likelihood weighted error: {lw_difference:.4f}")