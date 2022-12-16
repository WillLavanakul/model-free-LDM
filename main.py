from DQN import ReplayMemory, Transition, hidden_unit, Q_learning
from torch.autograd import Variable
from gridworld import *
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
from LDM import *

## Include the replay experience
state_func = initGrid_4x4
state_size = 64
nrow, ncol, _ = state_func().shape

def train_online(epochs, buffer, batch_size):
    gamma = 0.9 #since it may take several moves to goal, making gamma high
    epsilon = 1
    model = Q_learning(state_size, [150,150], 4, hidden_unit)
    optimizer = optim.Adam(model.parameters(), lr = 1e-2)
    # optimizer = optim.SGD(model.parameters(), lr = 0.1, momentum = 0)
    criterion = torch.nn.MSELoss()
    memory = ReplayMemory(buffer)   
    train_losses = []
    train_rewards = []

    for i in range(epochs):
        state = state_func()
        status = 1
        step = 0
        #while game still in progress
        running_reward = 0
        while(status == 1):   
            v_state = Variable(torch.from_numpy(state)).view(1,-1)
            qval = model(v_state)
            if (np.random.random() < epsilon): #choose random action
                action = np.random.randint(0,4)
            else: #choose best action from Q(s,a) values
                action = np.argmax(qval.data)
            #Take action, observe new state S'
            new_state = makeMove(state, action)
            step +=1
            v_new_state = Variable(torch.from_numpy(new_state)).view(1,-1)
            #Observe reward
            reward = getReward(new_state)
            running_reward += reward
            memory.push(v_state.data, action, v_new_state.data, reward)
            if (len(memory) < buffer): #if buffer not filled, add to it
                state = new_state
                if reward != -1: #if reached terminal state, update game status
                    break
                else:
                    continue
            transitions = memory.sample(batch_size)
            batch = Transition(*zip(*transitions))
            state_batch = Variable(torch.cat(batch.state))
            action_batch = Variable(torch.LongTensor(batch.action)).view(-1,1)
            new_state_batch = Variable(torch.cat(batch.new_state))
            reward_batch = Variable(torch.FloatTensor(batch.reward))
            non_final_mask = (reward_batch == -1)
            #Let's run our Q function on S to get Q values for all possible actions
            qval_batch = model(state_batch)
            # we only grad descent on the qval[action], leaving qval[not action] unchanged
            state_action_values = qval_batch.gather(1, action_batch)
            #Get max_Q(S',a)
            with torch.no_grad():
                newQ = model(new_state_batch)
            maxQ = newQ.max(1)[0]
    #         if reward == -1: #non-terminal state
    #             update = (reward + (gamma * maxQ))
    #         else: #terminal state
    #             update = reward + 0*maxQ
    #         y = reward_batch + (reward_batch == -1).float() * gamma *maxQ
            y = reward_batch
            y[non_final_mask] += gamma * maxQ[non_final_mask]
            y = y.view(-1,1)
            print("Game #: %s" % (i,), end='\r')
            loss = criterion(state_action_values, y)
            train_losses.append(loss.item())
            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            for p in model.parameters():
                p.grad.data.clamp_(-1, 1)
            optimizer.step()
            state = new_state
            if reward != -1:
                status = 0
            if step > 30:
                break
        train_rewards.append(running_reward)
        if epsilon > 0.1:
            epsilon -= (1/epochs)

    plt.scatter(range(len(train_losses)), train_losses, label='train losses')
    plt.legend()
    plt.title("Training losses over epochs")
    plt.savefig("online_loss")
    plt.clf()

    plt.scatter(range(len(train_rewards)), train_rewards, label='train rewards')
    plt.legend()
    plt.title("Training rewards over epochs")
    plt.savefig("online_rewards")
    plt.clf()
    return model, train_rewards

def train_offline(dataset, epochs, batch_size, ldm=0, ldm_g=1, ldm_c = 0.5, G=None, log_every=1):
    eval_trials = 10
    gamma = 0.9 #since it may take several moves to goal, making gamma high
    epsilon = 1
    model = Q_learning(state_size, [150,150], 4, hidden_unit)
    optimizer = optim.Adam(model.parameters(), lr = 1e-2)
    # optimizer = optim.SGD(model.parameters(), lr = 0.1, momentum = 0)
    criterion = torch.nn.MSELoss()
    train_losses = []
    eval_rewards = []
    eval_terms = []
    for i in range(epochs):
        print("Epoch #: {0}".format(i), end='\r')
        for b in range(len(dataset.memory) // batch_size):
            transitions = dataset.sample(batch_size)
            batch = Transition(*zip(*transitions))
            state_batch = Variable(torch.cat(batch.state))
            action_batch = Variable(torch.LongTensor(batch.action)).view(-1,1)
            new_state_batch = Variable(torch.cat(batch.new_state))
            reward_batch = Variable(torch.FloatTensor(batch.reward))
            non_final_mask = (reward_batch == -1)
            #Let's run our Q function on S to get Q values for all possible actions
            qval_batch = model(state_batch)
            # we only grad descent on the qval[action], leaving qval[not action] unchanged
            state_action_values = qval_batch.gather(1, action_batch)
            #Get max_Q(S',a)
            with torch.no_grad():
                newQ = model(new_state_batch)
            maxQ = newQ.max(1)[0]
            y = reward_batch
            
            if ldm == 0:
                y[non_final_mask] += gamma * maxQ[non_final_mask]
            elif ldm == 1:
                state_batch_val = state_batch_to_val(state_batch, nrow, ncol)
                action_batch_np = action_batch.numpy()
                G_sa = np.diagonal(G[state_batch_val.astype(int)][:, action_batch_np.astype(int)]).squeeze().copy()

                G_sa[(G_sa + np.log(ldm_c)) <= 0] = 0
                y[non_final_mask] += gamma * maxQ[non_final_mask] - ldm_g*G_sa[non_final_mask]
            y = y.view(-1,1)
            loss = criterion(state_action_values, y)
            train_losses.append(loss.item())
            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            train_losses.append(loss.item())
            for p in model.parameters():
                p.grad.data.clamp_(-1, 1)
            optimizer.step()
        if i % log_every == 0:
            eval_reward, eval_term = testAlgo(model, init=0, output=False, ldm=ldm, ldm_g=ldm_g, ldm_c=ldm_c, G=G)
            eval_rewards.append(eval_reward)
            eval_terms.append(eval_term)
    plt.scatter(range(len(train_losses)), train_losses, label='train losses')
    plt.legend()
    plt.title("Training losses over epochs")
    plt.savefig("offline_loss")
    plt.clf()

    plt.scatter(range(len(eval_rewards)), eval_rewards, label='train rewards')
    plt.legend()
    plt.title("Training rewards over epochs")
    plt.savefig("offline_rewards")
    plt.clf()
    return model, train_losses, eval_rewards, eval_terms


            

## Here is the test of AI
def testAlgo(model, init=0, output=True, ldm=0, ldm_g=1, ldm_c=0.5, G=None):
    i = 0
    state = state_func()
    if output:
        print("Initial State:")
        print(state[1, 1])
        print(dispGrid(state))
    status = 1
    #while game still in progress
    term = 0
    running_reward = 0
    while(status == 1):
        v_state = Variable(torch.from_numpy(state))
        qval = model(v_state.view(state_size))
        if ldm == 2:
            state_val = state_to_val(state)
            G_sa = torch.from_numpy(G[state_val].squeeze().copy())
            G_sa[(G_sa + np.log(ldm_c)) < 0] = 0
            qval = qval - ldm_g * G_sa
        elif ldm == 3:
            state_val = state_to_val(state)
            G_sa = torch.from_numpy(G[state_val].squeeze().copy())
            qval[(G_sa > -np.log(ldm_c)).reshape((1, 4))] = qval.min()-1
        action = np.argmax(qval.data) #take action with highest Q-value
        state = makeMove(state, action)
        if output:
            print('Move #: %s; Taking action: %s' % (i, action))
            print(dispGrid(state))
        reward = getReward(state)
        running_reward += reward
        if reward != -1:
            term = np.sign(reward).astype(int)
            print(term, reward)
            status = 0
            if output:
                print("Reward: %s" % (running_reward,))
        i += 1 #If we're taking more than 10 actions, just stop, we probably can't win this game
        if (i > 30):
            term = 0
            if output:
                print("Game lost; too many moves.")
            break
    return running_reward, term

def collect_data(model, n, epsilon):
    memory = ReplayMemory(n)  
    state = state_func()
    for i in range(n):
        v_state = Variable(torch.from_numpy(state)).view(1,-1)
        qval = model(v_state)
        if (np.random.random() < epsilon): #choose random action
            action = np.random.randint(0,4)
        else: #choose best action from Q(s,a) values
            action = np.argmax(qval.data)
        #Take action, observe new state S'
        new_state = makeMove(state, action)
        v_new_state = Variable(torch.from_numpy(new_state)).view(1,-1)
        #Observe reward
        reward = getReward(new_state)
        memory.push(v_state.data, action, v_new_state.data, reward)
        if (len(memory) < n): #if buffer not filled, add to it
                state = new_state
                if reward != -1: #if reached terminal state, update game status
                    state = state_func()
                else:
                    continue
    return memory
    
def offline_vis(dataset):
    state = state_func()
    nrow, ncol, _ = state.shape
    states = []
    state_heatmap = np.zeros((nrow, ncol))
    num_s = nrow*ncol
    num_a = 4
    for transition in dataset.memory:
        state = transition.state.reshape((nrow, ncol, 4))
        i, j = getLoc(state, 3)
        state_heatmap[i, j] += 1
        states.append([i, j])
    states = np.array(states)

    fig = plt.figure()
    plt.imshow(state_heatmap)
    plt.colorbar()
    plt.title('State Density')
    fig.savefig('dataset_state_density', bbox_inches='tight')
    plt.clf()

    plt.hist(states[:, 0]*nrow + states[:, 1], bins=4*4)
    plt.title("hist of states")
    fig.savefig('dataset_state_histogram', bbox_inches='tight')
    plt.clf()
    


if __name__ == "__main__":
    online_epochs = 1000
    online_buffer = 80
    online_batch_size = 40
    expert_model, _ = train_online(online_epochs, online_buffer, online_batch_size)
    testAlgo(expert_model, init=0)

    print("Collecting dataset...")
    offline_size = 1000
    epsilon = 0.9
    offline_dataset = collect_data(expert_model, offline_size, epsilon)
    offline_vis(offline_dataset)

    print("Evaluating LDM...")
    ldm_eval_g = 1
    density = compute_density(offline_dataset, nrow, ncol, 4)
    plt.imshow(density)
    plt.colorbar()
    plt.title('State Density')
    plt.savefig('state_action_density', bbox_inches='tight')
    plt.clf()
    G = define_G(state_func(), density, ldm_eval_g)

    print("Training offline model...")
    offline_epochs = 500
    offline_batch_size = 32
    k = 0.5
    ldm_g = 1
    ldm_c = get_c_from_threshold(density, offline_dataset, k)
    print(ldm_c)
    ldm = 1
    offline_model, train_losses, eval_rewards, eval_terms = train_offline(offline_dataset, offline_epochs, offline_batch_size, ldm=ldm, ldm_g=ldm_g, ldm_c=ldm_c, G=G)
    testAlgo(offline_model, init=0, ldm=ldm, ldm_g=ldm_g, ldm_c=ldm_c, G=G)