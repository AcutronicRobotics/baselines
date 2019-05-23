def mujoco():
    return dict(
        nsteps=2048,
        nminibatches=32,
        lam=0.95,
        gamma=0.99,
        noptepochs=10,
        log_interval=1,
        ent_coef=0.0,
        lr=lambda f: 3e-4 * f,
        cliprange=0.2,
        value_network='copy'
    )

def atari():
    return dict(
        nsteps=128, nminibatches=4,
        lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
        ent_coef=.01,
        lr=lambda f : f * 2.5e-4,
        cliprange=0.1,
    )
def retro():
   return atari()

def mara_mlp():
    return dict(
        num_layers = 2,
        num_hidden = 16,
        layer_norm = False,
        nsteps = 1024,
        nminibatches = 4, #batchsize = nevn * nsteps // nminibatches
        lam = 0.95,
        gamma = 0.99,
        noptepochs = 10,
        log_interval = 1,
        ent_coef = 0.0,
        lr = lambda f: 3e-3 * f,
        cliprange = 0.25,
        vf_coef = 1,
        max_grad_norm = 0.5,
        seed = 0,
        value_network = 'copy',
        network = 'mlp',
        total_timesteps = 1e8,
        save_interval = 10,
        env_name = 'MARA-v0',
        #env_name = 'MARAReal-v0',
        #env_name = 'MARAOrient-v0',
        # env_name = 'MARACollision-v0',
        # env_name = 'MARACollisionOrient-v0',
        transfer_path = None,
        # transfer_path = '/tmp/ros2learn/MARA-v0/ppo2_mlp/2019-02-19_12h47min/checkpoints/best',
        trained_path = '/tmp/ros2learn/MARA-v0/ppo2_mlp/2019-04-02_13h18min/checkpoints/best'
    )

def mara_lstm():
    return dict(
        nlstm = 256,
        layer_norm = False,
        # nbatch = nenvs * nsteps
        # nbatch_train = nbatch // nminibatches
        # assert nbatch % nminibatches == 0
        # assert batchsize == nbatch_train >= nsteps
        nsteps = 1024,
        #otherwise, last minibatch gets noisy gradient,
        nminibatches = 2, #batchsize = nevn * nsteps // nminibatches
        lam = 0.95,
        gamma = 0.99,
        noptepochs = 10,
        log_interval = 1,
        ent_coef = 0.0,
        lr = lambda f: 3e-4 * f,
        cliprange = 0.2,
        vf_coef = 0.5,
        max_grad_norm = 0.5,
        seed = 0,
        value_network = 'shared',
        network = 'lstm',
        total_timesteps = 1e8,
        save_interval = 10,
        env_name = 'MARARandomTarget-v0',
        num_envs = 8,
        transfer_path = None,
        # transfer_path = '/tmp/ros2learn/MARACollisionOrientRandomTarget-v0/ppo2_lstm/checkpoints/00090',
        trained_path = '/home/rkojcev/MARA_NN/lstm_server/best'
    )
