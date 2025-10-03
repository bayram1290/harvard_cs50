from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import numpy as np

print("=" * 60)
print("HIDDEN MARKOV MODEL - Bayesian network (Time slices)")
print("=" * 60)

def init_hmm_bayesian_network(steps: int):
    model = DiscreteBayesianNetwork()
    for i in range(steps):
        model.add_node(f'hidden_stat{i}')
        model.add_node(f'observe_stat{i}')
        model.add_edge(f'hidden_stat{i}', f'observe_stat{i}')
        if i > 0:
            model.add_edge(f'hidden_stat{i-1}', f'hidden_stat{i}')

    return model

observe_cnt = 9
hmm = init_hmm_bayesian_network(observe_cnt)

hidden_cpd0 = TabularCPD(
    variable='hidden_stat0',
    variable_card=2,
    values=[
        [0.5],
        [0.5]
    ],
    state_names={
        'hidden_stat0': ['sun', 'rain']
    }
)

for i in range(1, observe_cnt):
    transition_cpd = TabularCPD(
        variable=f'hidden_stat{i}',
        variable_card=2,
        values=[
            [0.8, 0.3],
            [0.2, 0.7]
        ],
        evidence=[f'hidden_stat{i-1}'],
        evidence_card=[2],
        state_names={
            f'hidden_stat{i}': ['sun', 'rain'],
            f'hidden_stat{i-1}': ['sun', 'rain'],
        }
    )
    hmm.add_cpds(transition_cpd)

for i in range(observe_cnt):
    observe_cpd = TabularCPD(
        variable=f'observe_stat{i}',
        variable_card=2,
        values=[
            [0.2, 0.9],
            [0.8, 0.1]
        ],
        evidence=[f'hidden_stat{i}'],
        evidence_card=[2],
        state_names={
            f'observe_stat{i}': ['umbrella', 'no umbrella'],
            f'hidden_stat{i}': ['sun', 'rain']
        }
    )
    hmm.add_cpds(observe_cpd)

hmm.add_cpds(hidden_cpd0)
print('HMM Bayesian Network model status: ', hmm.check_model())

print("\n" + "=" * 60)
print("VITERBI ALGORITHM FOR STATE PREDICTION")
print("=" * 60)

def viterbi(emission_probs, transition_probs, observations, initial_probs):
    states = list(initial_probs.keys())
    num_obs = len(observations)

    viterbi_table = np.zeros((len(states), num_obs))
    backpointers = np.zeros((len(states), num_obs), dtype=int)

    for i, state in enumerate(states):
        viterbi_table[i, 0] = (initial_probs[state] * emission_probs[state][observations[0]])

    for t in range(1, num_obs):
        for j, current_state in enumerate(states):
            max_prob = -1
            best_prev_state = 0

            for i, prev_state in enumerate(states):
                prob = (viterbi_table[i, t-1] * transition_probs[prev_state][current_state] * emission_probs[current_state][observations[t]])

                if prob > max_prob:
                    max_prob = prob
                    best_prev_state = i

            viterbi_table[j, t] = max_prob
            backpointers[j, t] = best_prev_state

    best_path = []
    best_final_state = np.argmax(viterbi_table[:, -1])
    best_path.append(states[best_final_state])

    for t in range(num_obs - 1, 0, -1):
        best_final_state = backpointers[best_final_state, t]
        best_path.append(states[best_final_state])

    return list(reversed(best_path))

initial_probs = {'sun': 0.5, 'rain': 0.5}
transition_probs = {
    'sun': {'sun': 0.8, 'rain': 0.2},
    'rain': {'sun': 0.3, 'rain': 0.7}
}
emission_probs = {
    'sun': {'umbrella': 0.2, 'no umbrella': 0.8},
    'rain': {'umbrella': 0.9, 'no umbrella': 0.1}
}
observations = [
    'umbrella',
    'umbrella',
    'no umbrella',
    'umbrella',
    'umbrella',
    'umbrella',
    'umbrella',
    'no umbrella',
    'no umbrella'
]

print("Observations: ", observations)
predictions = viterbi(emission_probs, transition_probs, observations, initial_probs)

print("\nPredicted hidden states:")
for i, (obs, pred) in enumerate(zip(observations, predictions)):
    print(f"Time {i}: Observation='{obs}', Predicted State='{pred}'")

print(f"\nFinal predictions: {predictions}")

print("\n" + "=" * 60)
print("FORWARD-BACKWARD ALGORITHM FOR POSTERIOR PROBABILITIES")
print("=" * 60)

def forward_backward(observations, initial_probs, transition_probs, emission_probs):
    states = list(initial_probs.keys())
    num_obs = len(observations)
    num_states = len(states)

    alpha = np.zeros((num_states, num_obs))

    for i, state in enumerate(states):
        alpha[i, 0] = initial_probs[state] * emission_probs[state][observations[0]]

    alpha[:, 0] /= np.sum(alpha[:, 0])

    for t in range(1, num_obs):
        for j, current_state in enumerate(states):
            alpha[j, t] = emission_probs[current_state][observations[t]]
            sum_val = 0
            for i, prev_state in enumerate(states):
                sum_val += alpha[i, t-1] * transition_probs[prev_state][current_state]
            alpha[j, t] *= sum_val

        alpha[:, t] /= np.sum(alpha[:, t])

    beta = np.ones((num_states, num_obs))

    for t in range(num_obs - 2, -1, -1):
        for i, current_state in enumerate(states):
            beta[i, t] = 0
            for j, next_state in enumerate(states):
                beta[i, t] += (beta[j, t+1] * transition_probs[current_state][next_state] * emission_probs[next_state][observations[t+1]])

        beta[:, t] /= np.sum(beta[:, t])

    posterior = alpha * beta
    posterior /= np.sum(posterior, axis=0)

    return posterior, alpha, beta

posterior_probs, alpha, beta = forward_backward(observations, initial_probs, transition_probs, emission_probs)

states_list = ['sun', 'rain']
print("\nPosterior probabilities P(state | observations):")
print("Time |  P(sun)  |  P(rain) | Most Likely")
print("-" * 45)
for t in range(len(observations)):
    sun_prob = posterior_probs[0, t]
    rain_prob = posterior_probs[1, t]
    most_likely = 'sun' if sun_prob > rain_prob else 'rain'
    print(f"{t:4} | {sun_prob:7.4f} | {rain_prob:7.4f} | {most_likely:>11}")


print("\n" + "=" * 60)
print("FINAL RESULTS COMPARISON")
print("=" * 60)

print('Viterbi (Most Likely Sequence): ' + ' ' * 8 , predictions)
print('Forward-backward (Per-time most likely):', ['sun' if posterior_probs[0, i] >= posterior_probs[1, i] else 'rain' for i in range(len(observations))])