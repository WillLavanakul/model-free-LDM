import matplotlib.pyplot as plt
import numpy as np
from main import *
from LDM import *

if __name__ == "__main__":
    ldm = 1

    online_epochs = 50
    online_buffer = 80
    online_batch_size = 40

    # high k -> less constrained to training dist
    k=0.5
    expert_model, train_rewards = train_online(online_epochs, online_buffer, online_batch_size)
    plt.plot(range(len(train_rewards)), train_rewards, label="Online DQN")
    plt.legend()
    plt.title("Expert policy rewards (Online DQN {0} epochs)".format(online_epochs))
    plt.savefig('exp/ldm{0}/eps_ldm{0}_expert_rew'.format(ldm))
    plt.clf()
    eps = [0.0, 0.2, 0.8, 1.0]
    num_offline = 10
    for ep in eps:
        print("Collecting dataset_eps{0}...".format(ep))
        offline_size = 500
        offline_dataset = collect_data(expert_model, offline_size, ep)

        print("Evaluating LDM_eps{0}...".format(ep))
        ldm_eval_g = 1
        density = compute_density(offline_dataset, 16, 4)
        plt.imshow(density)
        plt.title('(s,a) dataset density: eps={0}'.format(ep))
        plt.colorbar()
        plt.savefig('exp/ldm{0}/eps{1}_ldm{0}_data_density'.format(ldm, str(ep).replace('.', '')))
        plt.clf()

        G = define_G(initGrid(), density, ldm_eval_g)
        plt.imshow(G)
        plt.title('G(s,a): eps={0}'.format(ep))
        plt.colorbar()
        plt.savefig('exp/ldm{0}/eps{1}_ldm{0}_G'.format(ldm, str(ep).replace('.', '')))
        plt.clf()

        offline_epochs = 500
        offline_batch_size = 32
        ldm_g = 1
        ldm_c = get_c_from_threshold(density, offline_dataset, k)
        print(ldm_c)
        rewards_batch = []
        terms_batch = []
        for i in range(num_offline):
            print("Training offline {0}...".format(i))
            offline_model, train_losses, eval_rewards, eval_terms = train_offline(offline_dataset, offline_epochs, offline_batch_size, ldm=ldm, ldm_g=ldm_g, ldm_c=ldm_c, G=G)
            rewards_batch.append(eval_rewards)
            terms_batch.append(eval_terms)

        plt.figure(2)
        rewards_batch = np.array(rewards_batch).mean(axis=0)
        plt.scatter(range(len(rewards_batch)), rewards_batch, label='eps{0}_ldm{1}'.format(ep, ldm))
        
        plt.figure(3)
        terms_batch = np.array(terms_batch)
        percent_stall = (terms_batch == 0).sum(axis=0) / num_offline
        plt.scatter(range(len(percent_stall)), percent_stall, label='eps{0}_ldm{1}'.format(ep, ldm))

        plt.figure(4)
        percent_fail = (terms_batch == -1).sum(axis=0) / num_offline
        plt.scatter(range(len(percent_fail)), percent_fail, label='eps{0}_ldm{1}'.format(ep, ldm))

        plt.figure(5)
        percent_goal = (terms_batch == 1).sum(axis=0) / num_offline
        plt.scatter(range(len(percent_goal)), percent_goal, label='eps{0}_ldm{1}'.format(ep, ldm))
        plt.figure(1)

    plt.figure(2)
    plt.legend()
    plt.title('Average reward ldm{0}'.format(ldm))
    plt.savefig('exp/ldm{0}/eps_ldm{0}_rew'.format(ldm))

    plt.figure(3)
    plt.legend()
    plt.title('Percent stall (maxsteps 10) ldm{0}'.format(ldm))
    plt.savefig('exp/ldm{0}/eps_ldm{0}_stall'.format(ldm))

    plt.figure(4)
    plt.legend()
    plt.title('Percent fail ldm{0}'.format(ldm))
    plt.savefig('exp/ldm{0}/eps_ldm{0}_fail'.format(ldm))

    plt.figure(5)
    plt.legend()
    plt.title('Percent success ldm{0}'.format(ldm))
    plt.savefig('exp/ldm{0}/eps_ldm{0}_success'.format(ldm))



    