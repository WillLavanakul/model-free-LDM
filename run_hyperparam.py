from DQN import ReplayMemory, Transition, hidden_unit, Q_learning
from torch.autograd import Variable
from gridworld import *
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
from LDM import *
from main import *


def run_experiments_hyperparam():
    ldms = [0, 1, 2, 3]
    ldm_gs = [0.01, 0.1, 0.5, 1, 2, 5]
    ldm_cs = [0.01, 0.05, 0.1, 0.5, 0.8, 0.9]

    online_epochs = 500
    online_buffer = 80
    online_batch_size = 40
    expert_model = train_online(online_epochs, online_buffer, online_batch_size)

    print("Collecting dataset...")
    offline_size = 500
    offline_dataset = collect_data(expert_model, offline_size)

    print("Evaluating LDM...")
    ldm_eval_g = 1
    density = compute_density(offline_dataset, 16, 4)
    G = define_G(initGrid(), density, ldm_eval_g)
    plt.figure(figsize=(10,15))
    for ldm in ldms:
        print("ldm:", ldm)
        for ldm_g in ldm_gs:
            print("gamma:", ldm_g)
            for ldm_c in ldm_cs:
                print("ldm_c:", ldm_c)
                eval_returns = []
                for i in range(10):
                    print("Training offline model{0}...".format(i))
                    offline_epochs = 1000
                    offline_batch_size = 32
                    offline_model, train_losses, eval_rewards, eval_terms = train_offline(offline_dataset, offline_epochs, offline_batch_size, ldm=ldm, ldm_g=ldm_g, ldm_c=ldm_c, G=G)
                    eval_returns.append(eval_rewards)
                eval_returns = np.array(eval_returns)
                plt.plot(range(eval_returns.shape[1]), eval_returns.mean(axis=0), label='ldm{0}_g{1}_c{2}'.format(ldm, ldm_g, ldm_c))
    plt.title("4x4 Gridworld Offline E-LDM-DQN")
    plt.legend()
    plt.savefig('hyperparam')

if __name__ == "__main__":
    run_experiments_hyperparam()