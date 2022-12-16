import numpy as np

def randPair(s,e):
    return np.random.randint(s,e), np.random.randint(s,e)

#finds an array in the "depth" dimension of the grid
def findLoc(state, obj):
    nrow, ncol, _ = state.shape
    indices = []
    for i in range(0,nrow):
        for j in range(0,ncol):
            if (state[i,j] == obj).all():
                indices.append((i, j))
    return indices

#Initialize stationary grid, all items are placed deterministically
def initGrid_4x4():
    state = np.zeros((4,4,4))
    #place player
    state[0,1] = np.array([0,0,0,1])
    #place wall
    state[2,2] = np.array([0,0,1,0])
    #place pit
    state[1,1] = np.array([0,1,0,0])
    state[0,3] = np.array([0,1,0,0])
    state[3,0] = np.array([0,1,0,0])
    #place goal
    state[3,2] = np.array([1,0,0,0])
    return state

def initGrid_6x6():
    state = np.zeros((6,6,4))
    #place player
    state[0,1] = np.array([0,0,0,1])
    #place wall
    state[2,2] = np.array([0,0,1,0])
    state[3,5] = np.array([0,0,1,0])
    state[4,1] = np.array([0,0,1,0])
    #place pit
    state[1,1] = np.array([0,1,0,0])
    state[0,3] = np.array([0,1,0,0])
    # state[4,1] = np.array([0,1,0,0])
    state[1,5] = np.array([0,1,0,0])
    state[4,3] = np.array([0,1,0,0])
    state[3,0] = np.array([0,1,0,0])
    #place goal
    state[3,3] = np.array([1,0,0,0])
    return state

#Initialize player in random location, but keep wall, goal and pit stationary
def initGridPlayer():
    state = np.zeros((4,4,4))
    #place player
    state[randPair(0,4)] = np.array([0,0,0,1])
    #place wall
    state[2,2] = np.array([0,0,1,0])
    #place pit
    state[1,1] = np.array([0,1,0,0])
    #place goal
    state[1,2] = np.array([1,0,0,0])
    
    a = findLoc(state, np.array([0,0,0,1])) #find grid position of player (agent)
    w = findLoc(state, np.array([0,0,1,0])) #find wall
    g = findLoc(state, np.array([1,0,0,0])) #find goal
    p = findLoc(state, np.array([0,1,0,0])) #find pit
    if (not a or not w or not g or not p):
        #print('Invalid grid. Rebuilding..')
        return initGridPlayer()
    
    return state

#Initialize grid so that goal, pit, wall, player are all randomly placed
def initGridRand():
    state = np.zeros((4,4,4))
    #place player
    state[randPair(0,4)] = np.array([0,0,0,1])
    #place wall
    state[randPair(0,4)] = np.array([0,0,1,0])
    #place pit
    state[randPair(0,4)] = np.array([0,1,0,0])
    #place goal
    state[randPair(0,4)] = np.array([1,0,0,0])
    
    a = findLoc(state, np.array([0,0,0,1]))
    w = findLoc(state, np.array([0,0,1,0]))
    g = findLoc(state, np.array([1,0,0,0]))
    p = findLoc(state, np.array([0,1,0,0]))
    #If any of the "objects" are superimposed, just call the function again to re-place
    if (not a or not w or not g or not p):
        #print('Invalid grid. Rebuilding..')
        return initGridRand()
    
    return state

def set_state(state, s):
    nrow, ncol, _ = state.shape
    new_i, new_j = s // nrow, s % ncol
    i, j = getLoc(state, 3)
    if (state[new_i, new_j] == 0).all():
        state[new_i, new_j] = np.array([0,0,0,1])
        state[i, j] = np.array([0, 0, 0, 0])
    return state

def state_to_val(state):
    nrow, ncol, _ = state.shape
    i, j = getLoc(state.reshape((nrow, ncol, 4)), 3)
    state_val = i*nrow+j
    return state_val

def state_batch_to_val(state, nrow, ncol):
    state_vals = []
    for s in state:
        s = s.reshape((nrow, ncol, 4))
        i, j = getLoc(s, 3)
        state_val = i*nrow + j
        state_vals.append(state_val)
    return np.array(state_vals)

def makeMove(state, action):
    #need to locate player in grid
    #need to determine what object (if any) is in the new grid spot the player is moving to
    nrow, ncol, _ = state.shape
    player_loc = findLoc(state, np.array([0,0,0,1]))[0]
    wall = findLoc(state, np.array([0,0,1,0]))
    goal = findLoc(state, np.array([1,0,0,0]))
    pit = findLoc(state, np.array([0,1,0,0]))
    state = np.zeros((nrow,ncol,4))

    actions = [[-1,0],[1,0],[0,-1],[0,1]]
    #e.g. up => (player row - 1, player column + 0)
    new_loc = (player_loc[0] + actions[action][0], player_loc[1] + actions[action][1])
    if (new_loc not in wall):
        if ((np.array(new_loc) <= (3,3)).all() and (np.array(new_loc) >= (0,0)).all()):
            state[new_loc][3] = 1

    new_player_loc = findLoc(state, np.array([0,0,0,1]))
    #print(new_player_loc)
    if (len(new_player_loc) == 0):
        state[player_loc] = np.array([0,0,0,1])
    #re-place pit
    for i,j in pit:
        state[i,j][1] = 1
    #re-place wall
    for i,j in wall:
        state[i,j][2] = 1
    #re-place goal
    for i,j in goal:
        state[i,j][0] = 1

    return state

def getLoc(state, level):
    nrow, ncol, _ = state.shape
    for i in range(0,nrow):
        for j in range(0,ncol):
            if (state[i,j][level] == 1):
                return i,j

def getReward(state):
    player_loc = getLoc(state, 3)
    goal = getLoc(state, 0)
    pit = findLoc(state, np.array([0,1,0,1]))
    if (player_loc in pit):
        return -30
    elif (player_loc == goal):
        return 30
    else:
        return -1
    
def dispGrid(state):
    nrow, ncol, _ = state.shape
    grid = np.zeros((nrow,ncol), dtype= str)
    for i in range(0,nrow):
        for j in range(0,ncol):
            if (state[i, j] == np.array([0,0,0,1])).all():
                grid[i,j] = 'P'
            elif (state[i, j] == np.array([0,0,1,0])).all():
                grid[i,j] = 'W'
            elif (state[i, j] == np.array([1,0,0,0])).all():
                grid[i,j] = '+'
            elif (state[i, j] == np.array([0,1,0,0])).all():
                grid[i,j] = 'O'
            else:
                grid[i,j] = '_'
    
    return grid