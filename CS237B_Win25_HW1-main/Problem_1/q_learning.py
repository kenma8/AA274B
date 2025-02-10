import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils import generate_problem, visualize_value_function

from value_iteration import value_iteration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def Q_learning(Q_network, reward_fn, is_terminal_fn, X, U, Xp, gam):
    assert X.ndim == 2 and U.ndim == 2 and Xp.ndim == 2
    sdim, adim = X.shape[-1], U.shape[-1]

    def loss():
        batch_n = int(1e4)
        ridx = torch.randint(0, X.shape[0], (batch_n,), device=device)
        X_, U_, Xp_ = X[ridx], U[ridx], Xp[ridx]

        U_all = torch.arange(4, dtype=torch.float32, device=device).view(1, -1, 1).repeat(batch_n, 1, 1)
        Xp_all = Xp_.unsqueeze(1).repeat(1, 4, 1)
        U_all = U_all.reshape(-1, 1)
        Xp_all = Xp_all.reshape(-1, sdim)
        input_next = torch.cat([Xp_all, U_all], -1)
        next_Q = Q_network(input_next).reshape(batch_n, 4).max(-1)[0] 

        input_current = torch.cat([X_, U_], -1)
        Q = Q_network(input_current).reshape(-1)

        ######### Your code starts here #########
        # given the current (Q) and the optimal next state Q function (Q_next), 
        # compute the Q-learning loss

        # make sure to account for the reward, the terminal state and the
        # discount factor gam

        target_Q = reward_fn(X_, U_)  + gam * next_Q * (~is_terminal_fn(X_))
        l = torch.mean((Q - target_Q) ** 2)
    
        ######### Your code ends here ###########

        # need to regularize the Q-value, because we're training its difference
        l = l + 1e-3 * torch.mean(Q ** 2)
        return l

    ######### Your code starts here #########
    # create the Adam optimizer with pytorch 
    # experiment with different learning rates [1e-4, 1e-3, 1e-2, 1e-1]
    learning_rate = 1e-3
    optimizer = optim.Adam(Q_network.parameters(), lr=learning_rate)

    ######### Your code ends here ###########

    print("Training the Q-network")
    for _ in tqdm(range(int(1e4))):
        ######### Your code starts here #########
        # apply a single step of gradient descent to the Q_network variables
        # take a look at the torch.optim module

        optimizer.zero_grad()
        l = loss()
        l.backward()
        optimizer.step()

        ######### Your code ends here ###########

# Q-learning ##################################################################
def main():
    
    problem = generate_problem()
    n = problem["n"]
    sdim, adim = n * n, 1
    Ts = problem["Ts"]  # Transition matrices
    idx2pos = torch.tensor(problem["idx2pos"], dtype=torch.float32)

    # Sample state-action triples
    samp_nb = int(1e5)
    try:
        with open("state_transitions.pkl", "rb") as fp:
            temp = pickle.load(fp)
            X, U, Xp = [torch.tensor(z, dtype=torch.float32) for z in temp]
    except FileNotFoundError:
        X = torch.randint(0, sdim, (samp_nb,))
        U = torch.randint(0, 4, (samp_nb,))
        x_list, u_list, xp_list = [], [], []
        print("Sampling state transitions")
        for i in tqdm(range(samp_nb)):
            x = X[i]
            u = U[i]
            ######### Your code starts here #########
            # x is the integer state index in the vectorized state shape: []
            # u is the integer action shape: []
            # compute xp, the integer next state shape: []

            # Hints: 
            # 1. make use of the transition matrices and torch.multinomial
            # 2. remember that transition matrices have a shape [sdim, sdim]
            # 3. remember that torch.multinomial takes in the probabilities, not the log-probabilities
            # 4. torch.squeeze() can be used to reduce a dimension of the tensor

            weights = Ts[u][x]
            xp = torch.multinomial(weights, 1).item()

            ######### Your code ends here ###########

            # convert integer states to a 2D representation using idx2pos
            xp = torch.tensor(idx2pos[xp])
            x_list.append(idx2pos[x].tolist())
            u_list.append([float(u)])
            xp_list.append(xp.tolist())

        X, U, Xp = torch.tensor(x_list), torch.tensor(u_list), torch.tensor(xp_list)
        with open("state_transitions.pkl", "wb") as fp:
            pickle.dump((X.numpy(), U.numpy(), Xp.numpy()), fp)

    # define the reward ####################################
    reward_vec = np.zeros([sdim])
    reward_vec[problem["pos2idx"][(19, 9)]] = 1.0
    reward_vec = torch.tensor(reward_vec, dtype=torch.float32)

    def reward_fn(X, U):
        return (X == torch.tensor([19.0, 9.0])).all(dim=-1).float()

    def is_terminal_fn(X):
        return (X == torch.tensor([19.0, 9.0])).all(dim=-1)

    ######### Your code starts here #########
    # create the deep Q-network using torch.nn
    # it needs to take in 2 state + 1 action input (3 inputs)
    # it needs to output a single value (batch x 1 output) - the Q-value
    # it should have 3 dense layers () with a width of 64 (two hidden 64 neuron embeddings)

    class QNetwork(nn.Module):
        def __init__(self):
            super(QNetwork, self).__init__()

            self.stack = nn.Sequential(
                nn.Linear(3, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )

        def forward(self, x):
            return self.stack(x)
        
    ######### Your code ends here ###########
    Q_network = QNetwork().to(device)

    # Train the Q-network
    gam = 0.95
    Q_learning(Q_network, reward_fn, is_terminal_fn, X, U, Xp, gam)

    ########################################################

    # Visualize the Q-network
    y, x = [
        torch.tensor(z, dtype=torch.float32, device=device).reshape(-1)
        for z in np.meshgrid(np.arange(n), np.arange(n))
    ]
    X_ = torch.arange(n * n, device=device)
    X_ = torch.stack([x[X_], y[X_]], -1).unsqueeze(1).repeat(1, 4, 1)

    U_ = torch.arange(4, dtype=torch.float32, device=device).view(1, -1, 1).repeat(sdim, 1, 1)
    X_, U_ = X_.reshape(-1, 2), U_.reshape(-1, 1)
    q_input = torch.cat([X_, U_], -1)

    with torch.no_grad():
        Q = Q_network(q_input).reshape(-1, 4)
        V = Q.max(-1)[0]
    
    # Visualize the result
    plt.figure(120)
    visualize_value_function(V.cpu().numpy().reshape((n, n)))
    plt.colorbar()
    plt.show()

    # create the terminal mask vector
    terminal_mask = np.zeros([sdim])
    terminal_mask[problem["pos2idx"][19, 9]] = 1.0
    terminal_mask = torch.tensor(terminal_mask, dtype=torch.float32)

    # generate the reward vector
    reward = np.zeros([sdim, 4])
    reward[problem["pos2idx"][19, 9], :] = 1.0
    reward = torch.tensor(reward, dtype=torch.float32)

    V_opt, V_policy = value_iteration(problem, reward, terminal_mask, gam)

    Q_policy = Q.argmax(-1)

    # create a binary heatmap plot that shows, for every state, if the approximate Q-network policy agrees or disagrees with the value iteration optimal policy
    plt.figure(121)
    plt.imshow((Q_policy == V_policy).cpu().numpy().reshape((n, n)).T, origin="lower")
    plt.title("Q-network policy agreement with value iteration")
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()
