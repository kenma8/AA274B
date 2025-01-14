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

        ######### Your code ends here ###########

        if err < 1e-7:
            break

    return V

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
    V_opt = value_iteration(problem, reward, terminal_mask, gam)

    # Visualize the Value Function
    plt.figure(213)
    visualize_value_function(V_opt.numpy().reshape((n, n)))
    plt.title("value iteration")
    plt.show()

if __name__ == "__main__":
    main()
