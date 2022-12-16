import matplotlib.pyplot as plt
import numpy as np
from main import *
from LDM import *

if __name__ == "__main__":
    ldm = 1

    online_epochs = 150
    online_buffer = 80
    online_batch_size = 40
    state_func = initGrid_4x4
    state_size = '4x4'
    state_size = 64
    nrow, ncol, _ = state_func().shape

    # high k -> less constrained to training dist
    k_s=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    expert_model, train_rewards = train_online(online_epochs, online_buffer, online_batch_size)
    testAlgo(expert_model)
    plt.plot(range(len(train_rewards)), train_rewards, label="Online DQN")
    plt.legend()
    plt.title("Expert policy rewards (Online DQN {0} epochs)".format(online_epochs))
    plt.savefig('exp/k_study/expert_rew_{0}'.format(state_size))
    plt.clf()
    ep = 0.2
    num_offline = 10
    print("Collecting dataset_eps{0}...".format(ep))
    offline_size = 120
    offline_dataset = collect_data(expert_model, offline_size, ep)

    print("Evaluating LDM_eps{0}...".format(ep))
    ldm_eval_g = 1
    density = compute_density(offline_dataset, nrow, ncol, 4)
    plt.imshow(density)
    plt.title('(s,a) dataset density: eps={0}'.format(ep))
    plt.colorbar()
    plt.savefig('exp/k_study/data_density_{0}'.format(state_size))
    plt.clf()

    G = define_G(state_func(), density, ldm_eval_g)
    plt.imshow(G)
    plt.title('G(s,a): eps={0}'.format(ep))
    plt.colorbar()
    plt.savefig('exp/k_study/G_{0}'.format(state_size))
    plt.clf()


    for k in k_s:
        ldm_c = get_c_from_threshold(density, offline_dataset, k)
        print("k:", k)
        print("ldm_c:", ldm_c)
        k_string = str(k).replace('.', '')

        for ldm in range(4):
            offline_epochs = 500
            offline_batch_size = 32
            ldm_g = 3
            if ldm == 2:
                ldm_g = 0.1
            log_every = 25
            rewards_batch = []
            terms_batch = []
            for i in range(num_offline):
                print("Training offline ldm{1} model{0}...".format(i, ldm))
                offline_model, train_losses, eval_rewards, eval_terms = train_offline(offline_dataset, offline_epochs, offline_batch_size, ldm=ldm, ldm_g=ldm_g, ldm_c=ldm_c, G=G, log_every=log_every)
                rewards_batch.append(eval_rewards)
                terms_batch.append(eval_terms)
            
            t = np.linspace(0, offline_epochs, offline_epochs // log_every)
            plt.figure(2)
            rewards_batch = np.array(rewards_batch).mean(axis=0)
            plt.plot(t, rewards_batch, label='ldm{1}'.format(ep, ldm))
            
            plt.figure(3)
            terms_batch = np.array(terms_batch)
            percent_stall = (terms_batch == 0).sum(axis=0) / num_offline
            plt.plot(t, percent_stall, label='ldm{1}'.format(ep, ldm))

            plt.figure(4)
            percent_fail = (terms_batch == -1).sum(axis=0) / num_offline
            plt.plot(t, percent_fail, label='ldm{1}'.format(ep, ldm))

            plt.figure(5)
            percent_goal = (terms_batch == 1).sum(axis=0) / num_offline
            plt.plot(t, percent_goal, label='ldm{1}'.format(ep, ldm))
            plt.figure(1)

        plt.figure(2)
        plt.legend()
        plt.title('Average reward eps{1} k{0}'.format(k_string, ep))
        plt.savefig('exp/k_study/k{0}/rew_{1}'.format(k_string, state_size))
        plt.clf()

        plt.figure(3)
        plt.legend()
        plt.title('Percent stall (maxsteps 10) eps{1} k{0}'.format(k_string, ep))
        plt.savefig('exp/k_study/k{0}/stall_{1}'.format(k_string, state_size))
        plt.clf()

        plt.figure(4)
        plt.legend()
        plt.title('Percent fail eps{1} k{0}'.format(k_string, ep))
        plt.savefig('exp/k_study/k{0}/fail_{1}'.format(k_string, state_size))
        plt.clf()

        plt.figure(5)
        plt.legend()
        plt.title('Percent success eps{1} k{0}'.format(k_string, ep))
        plt.savefig('exp/k_study/k{0}/success_{1}'.format(k_string, state_size))
        plt.clf()



    