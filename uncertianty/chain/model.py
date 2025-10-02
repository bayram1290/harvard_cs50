from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
import numpy as np
import ast
from collections import Counter


print("=" * 60)
print("MARKOV CHAIN - SAMPLING")
print("=" * 60)


def markov_chain_sample(initial_probs, transition_matrix, num_samples):
    states = list(initial_probs.keys())
    initial_probs_list = [initial_probs[state] for state in states]

    current_state = np.random.choice(states, p=initial_probs_list)
    samples = [current_state]

    for _ in range(num_samples - 1):
        next_state_probs = [transition_matrix[current_state][next_state] for next_state in states]
        current_state = np.random.choice(states, p=next_state_probs)
        samples.append(current_state)

    return samples

initial_probs = {'sun': 0.5, 'rain': 0.5}
transition_matrix = {
    'sun': {'sun': 0.8, 'rain': 0.2},
    'rain': {'sun': 0.3, 'rain': 0.7}
}

samples = markov_chain_sample(initial_probs, transition_matrix, 50)
print("Samples from Markov Chain:")
print([str(x) for x in samples])
print(f"Number of samples: {len(samples)}")

print("\n" + "=" * 60)
print("MARKOV CHAIN - BAYESIAN NETWORK")
print("=" * 60)

model = DiscreteBayesianNetwork()

time_steps = 5
for i in range(time_steps):
    model.add_node(f'X{i}')

for i in range(time_steps - 1):
    model.add_edge(f'X{i}', f'X{i+1}')

cpd_x0 = TabularCPD(
    variable='X0',
    variable_card=2,
    values=[[0.5], [0.5]],
    state_names={'X0': ['sun', 'rain']}
)

cpd_transition = TabularCPD(
    variable='X1',
    variable_card=2,
    values=[
        [0.8, 0.3],
        [0.2, 0.7]
    ],
    evidence=['X0'],
    evidence_card=[2],
    state_names={
        'X1': ['sun', 'rain'],
        'X0': ['sun', 'rain']
    }
)

model.add_cpds(cpd_x0)

for i in range(1, time_steps):
    cpd = TabularCPD(
        variable=f'X{i}',
        variable_card=2,
        values=[
            [0.8, 0.3],
            [0.2, 0.7]
        ],
        evidence=[f'X{i-1}'],
        evidence_card=[2],
        state_names={
            f'X{i}': ['sun', 'rain'],
            f'X{i-1}': ['sun', 'rain']
        }
    )
    model.add_cpds(cpd)

print("Bayesian Network model check:", model.check_model())

sampler = BayesianModelSampling(model)
samples_df = sampler.forward_sample(size=1)

print("\nSingle sequence from Bayesian Network (5 time steps):")
print(samples_df)

print("\n" + "=" * 60)
print("MULTIPLE SEQUENCES FROM BAYESIAN NETWORK")
print("=" * 60)


multiple_sequences = sampler.forward_sample(size=10)
print("10 sequences (rows) of 5 time steps (columns):")
print(multiple_sequences)

print("\n" + "=" * 60)
print("EXTENDED SAMPLING FOR 50 STATES")
print("=" * 60)


long_model = DiscreteBayesianNetwork()
for i in range(50):
    long_model.add_node(f'T{i}')

for i in range(49):
    long_model.add_edge(f'T{i}', f'T{i+1}')

long_model.add_cpds(TabularCPD(
    variable='T0',
    variable_card=2,
    values=[[0.5], [0.5]],
    state_names={'T0': ['sun', 'rain']}
))

for i in range(1, 50):
    long_model.add_cpds(TabularCPD(
        variable=f'T{i}',
        variable_card=2,
        values=[[0.8, 0.3], [0.2, 0.7]],
        evidence=[f'T{i-1}'],
        evidence_card=[2],
        state_names={
            f'T{i}': ['sun', 'rain'],
            f'T{i-1}': ['sun', 'rain']
        }
    ))

print("Long model check:", long_model.check_model())
sampler_long = BayesianModelSampling(long_model)
long_sequence_df = sampler_long.forward_sample(size=1)

long_sequence = long_sequence_df.iloc[0].tolist()
print(f"\nSingle sequence of 50 states from long chain:")
print(Counter(long_sequence))
print(long_sequence)
print(f"Number of states: {len(long_sequence)}")










print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)

print("Simple function samples (first 10):")
print([str(x) for x in samples[:10]])

print("\nBayesian Network samples (first 10):")
print(long_sequence[:10])


print("\n" + "=" * 60)
print("DISTRIBUTION ANALYSIS")
print("=" * 60)

sun_count_simple = samples.count('sun')
rain_count_simple = samples.count('rain')
sun_count_bn = long_sequence.count('sun')
rain_count_bn = long_sequence.count('rain')

print("Simple function distribution:")
print(f"   Sun: {sun_count_simple}/50 ({sun_count_simple/50*100:.1f}%)")
print(f"   Rain: {rain_count_simple}/50 ({rain_count_simple/50*100:.1f}%)")

print("\nBayesian Network distribution:")
print(f"  Sun: {sun_count_bn}/50 ({sun_count_bn/50*100:.1f}%)")
print(f"  Rain: {rain_count_bn}/50 ({rain_count_bn/50*100:.1f}%)")

stationary_sun = (1 - 0.7) / (2 - 0.8 - 0.7)
stationary_rain = 1 - stationary_sun
print(f"\nTheoretical stationary distribution:")
print(f"  Sun: {stationary_sun:.3f} ({stationary_sun*100:.1f}%)")
print(f"  Rain: {stationary_rain:.3f} ({stationary_rain*100:.1f}%)")
