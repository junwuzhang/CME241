from typing import TypeVar, Tuple, Mapping

# Psuedo-code

def policyEvaluation(mdp : MDP, policy : Policy) -> Mapping[S, float]:
    tmp_value = 0
    value_func = {}
    for s in mdp.non_terminal_states:
        for action, probability in policy[s]:
            for next_state in mdp.transition_data:
                tmp_value += mdp.transition_data[s][action][next_state] * mdp.state_value[s] * mdp.discount_factor + mdp.reward_data[s][a]
                value_func[s] = tmp_value
    return value_func
