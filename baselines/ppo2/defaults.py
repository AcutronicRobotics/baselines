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
        cliprange=lambda f : f * 0.1,
    )
def retro():
   return atari()

def mara_mlp():
    return dict(
        num_layers = 4,
        num_hidden = 128,
        layer_norm = False,
        nsteps = 2048,
        nminibatches = 32,
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
        value_network = 'copy',
        network = 'mlp',
        total_timesteps = 1e8,
        save_interval = 10,
        env_name = 'MARACollision-v0',
        num_envs = 2
    )

def mara_lstm():
    return dict(
        # nbatch = nenvs * nsteps
        # nbatch_train = nbatch // nminibatches
        # assert nbatch % nminibatches == 0
        # assert batchsize == nbatch_train >= nsteps
        # nan < 1024
        # gpu > 512
        nsteps = 1024,
        #otherwise, last minibatch gets noisy gradient,
        # careful this by default is 1, please change it in your script
        nminibatches = 2, #batchsize = nevn * nsteps // nminibatches
        lam = 0.95,
        gamma = 0.99,
        noptepochs = 10,
        log_interval = 1,
        ent_coef = 0.0,
        lr = lambda f: 3e-4 * f,
        cliprange = 0.2,
        vf_coef = 0.5,
        seed = 0,
        max_grad_norm = 0.5,
        value_network = 'shared',
        network = 'lstm',
        nlstm = 256,
        layer_norm = False,
        total_timesteps = 1e8,
        save_interval = 10
    )
