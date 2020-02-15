from typing import TypeVar, Tuple, Mapping

S = TypeVar('S')
A = TypeVar('A')

# Psuedo-code

def val_iteration(mdp: MDP) -> Mapping[S, float]:
    iteration_counter = 0
    iteration_limit = 1000
    final_value_function = mdp.action_value_functions
    action_value = {}
    while iteration_counter <= iteration_limit:
        for s in mdp.non_terminal_states:
            for a in mdp.actions:
                # action value function for action A
                action_value[a] = mdp.reward_data[a][1] * mdp.reward_data[a][0] \
                             + mdp.discount_factor * (mdp.reward_data[a][1] * final_value_function[mdp.transition_data[s][a]])
            best_value = max(action_value.items())
            if best_value > final_value_function[s]:
                final_value_function[s] = best_value
            iteration_counter += 1 

    return final_value_function