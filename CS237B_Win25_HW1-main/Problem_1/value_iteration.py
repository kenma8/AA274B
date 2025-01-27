import torch
import numpy as np
import matplotlib.pyplot as plt

from utils import generate_problem, visualize_value_function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def value_iteration(problem, reward, terminal_mask, gam):
    Ts = problem["Ts"]
    sdim, adim = Ts[0].shape[-1], len(Ts)  # state and action dimension
    V = torch.zeros([sdim], device = device)

    assert terminal_mask.ndim == 1 and reward.ndim == 2

    # perform value iteration
    for _ in range(1000):
        ######### Your code starts here #########
        # perform the value iteration update
        # V has shape [sdim]; sdim = n * n is the total number of grid states
        # Ts is a 4-element python list of transition matrices for 4 actions

        # reward has shape [sdim, 4] - represents the reward for each state-action pair

        # terminal_mask has shape [sdim] and has entries 1 for terminal states

        # compute the next value function estimate for the iteration
        # compute err = torch.linalg.norm(V_new - V_prev) as a breaking condition

        V_new = torch.zeros([sdim], device = device)
        for u in range(adim):
            V_new = torch.maximum(V_new, reward[:, u] + (gam * Ts[u] @ V) * (1 - terminal_mask))
        err = torch.linalg.norm(V_new - V)
        V = V_new

        ######### Your code ends here ###########

        if err < 1e-7:
            break

    policy = torch.zeros([sdim], dtype=torch.long, device=device)
    
    for s in range(sdim):
        if terminal_mask[s]: 
            continue
        
        Q_values = torch.zeros([adim], device=device)
        for u in range(adim):
            Q_values[u] = reward[s, u] + gam * (Ts[u][s] @ V)
        
        policy[s] = torch.argmax(Q_values) 

    return V, policy

def simulate_MDP(policy, problem, n, num_steps):
    Ts = problem["Ts"]
    sdim, adim = Ts[0].shape[-1], len(Ts)
    pos2idx = problem["pos2idx"]

    s = pos2idx[0, 0]
    states = [s]

    for _ in range(num_steps):
        a = policy[s]
        transition_probabilities = Ts[a][s].cpu().numpy()
        transition_probabilities /= transition_probabilities.sum()
        s = np.random.choice(sdim, p=transition_probabilities)
        states.append(s)
        if s == pos2idx[19, 9]:
            break

    states = np.array(states)
    X, Y = np.unravel_index(states, (n, n))

    return X, Y

# value iteration ##############################################################
def main():
    # generate the problem
    problem = generate_problem()
    n = problem["n"]
    sdim, adim = n * n, 1

    # create the terminal mask vector
    terminal_mask = np.zeros([sdim])
    terminal_mask[problem["pos2idx"][19, 9]] = 1.0
    terminal_mask = torch.tensor(terminal_mask, dtype=torch.float32)

    # generate the reward vector
    reward = np.zeros([sdim, 4])
    reward[problem["pos2idx"][19, 9], :] = 1.0
    reward = torch.tensor(reward, dtype=torch.float32)

    gam = 0.95
    V_opt, policy = value_iteration(problem, reward, terminal_mask, gam)
    simulated_trajectory = simulate_MDP(policy, problem, n, 100)

    # Visualize the Value Function
    plt.figure(213)
    visualize_value_function(V_opt.numpy().reshape((n, n)), arrows=True)
    plt.title("value iteration")
    plt.show()

    plt.figure(214)
    visualize_value_function(policy.numpy().reshape((n, n)), arrows=False)
    plt.plot(simulated_trajectory[0], simulated_trajectory[1], "ro-")

    plt.title("simulated trajectory")
    plt.show()

    '''
    # Visualize the simulated trajectory
    plt.figure(214)
    plt.title("simulated trajectory")
    plt.show()
    '''


if __name__ == "__main__":
    main()
