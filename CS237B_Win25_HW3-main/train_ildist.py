import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from utils import load_data, maybe_makedirs
import torch.distributions as D

class NN(nn.Module):
    def __init__(self, in_size, out_size):
        super(NN, self).__init__()
        
        ######### Your code starts here #########
        # We want to define and initialize the weights & biases of the neural network.
        # - in_size is dim(O)
        # - out_size is dim(A) = 2
        # HINT: You can use either of the following for weight initialization:
        #         - nn.init.xavier_uniform_()
        #         - nn.init.kaiming_uniform_()

        self.fc1 = nn.Linear(in_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 5)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)

        ########## Your code ends here ##########

    def forward(self, x):
        x = x.float()
        ######### Your code starts here #########
        # We want to perform a forward-pass of the network.
        # x is a (?, |O|) tensor that keeps a batch of observations

        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        
        return x
        ########## Your code ends here ##########


   
def loss_fn(y_est, y):
    """
    Calculate the negative log likelihood loss.
    
    y_est: Output of the network, where the first two columns represent the
           mean vector and the remaining four are covariance parameters.
    y: Target actions taken by the expert.
    """
    y = y.float()
    ######### Your code starts here #########
    # We want to compute the negative log-likelihood loss between y_est and y where
    # - y_est is the output of the network for a batch of observations,
    # - y is the actions the expert took for the corresponding batch of observations
    # At the end your code should return the scalar loss value.
    # HINT: You may find some of the following functions useful, but feel free to use your own implementation:
    #       - D.MultivariateNormal()
    #       - torch.diag_embed()
    #       - torch.bmm()
    #       - F.softplus()
    
    mu = y_est[:, :2]
    c = y_est[:, 2:]

    A = torch.zeros((y_est.shape[0], 2, 2))
    A[:, 0, 0] = F.softplus(c[:, 0])
    A[:, 1, 1] = F.softplus(c[:, 2])
    A[:, 1, 0] = c[:, 1]

    cov = torch.bmm(A, A.transpose(1, 2))

    dist = D.MultivariateNormal(mu, cov)

    loss = -dist.log_prob(y)
    return loss.mean()
    ########## Your code ends here ##########


def train_model(data, args):
    params = {
        'train_batch_size': 4096*32,
    }

    x_train = torch.tensor(data['x_train'], dtype=torch.float32)
    y_train = torch.tensor(data['y_train'], dtype=torch.float32)
    in_size = x_train.shape[-1]
    out_size = y_train.shape[-1]

    model = NN(in_size, out_size)
    if args.restore:
        model.load_state_dict(torch.load('./policies/' + args.scenario.lower() + '_' + args.goal.lower() + '_ILDIST.pt'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=params['train_batch_size'], shuffle=True)

    model.train()
    for epoch in range(args.epochs):
        train_loss = 0.0
        count = 0
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            ######### Your code starts here #########
            # HINT: This section is very similar to the section in train_il.py

            y_est = model(x_batch)
            loss = loss_fn(y_est, y_batch)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ########## Your code ends here ##########
            count += 1

        avg_loss = train_loss / count if count > 0 else 0.0
        print(f"Epoch {epoch + 1}, Loss: {avg_loss}")

    torch.save(model.state_dict(), './policies/' + args.scenario.lower() + '_' + args.goal.lower() + '_ILDIST.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--goal', type=str, help="left, straight, right, inner, outer, all", default="all")
    parser.add_argument('--scenario', type=str, help="intersection, circularroad", default="intersection")
    parser.add_argument("--epochs", type=int, help="number of epochs for training", default=1000)
    parser.add_argument("--lr", type=float, help="learning rate for Adam optimizer", default=1e-3)
    parser.add_argument("--restore", action="store_true", default=False)
    args = parser.parse_args()
    
    maybe_makedirs("./policies")
    
    data = load_data(args)
    train_model(data, args)
