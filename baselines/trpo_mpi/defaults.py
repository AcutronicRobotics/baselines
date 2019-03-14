from baselines.common.models import mlp, cnn_small

def atari():
    return dict(
        network = cnn_small(),
        timesteps_per_batch=512,
        max_kl=0.001,
        cg_iters=10,
        cg_damping=1e-3,
        gamma=0.98,
        lam=1.0,
        vf_iters=3,
        vf_stepsize=1e-4,
        entcoeff=0.00,
    )

def mujoco():
    return dict(
        network = mlp(num_hidden=32, num_layers=2),
        timesteps_per_batch=1024,
        max_kl=0.01,
        cg_iters=10,
        cg_damping=0.1,
        gamma=0.99,
        lam=0.98,
        vf_iters=5,
        vf_stepsize=1e-3,
        normalize_observations=True,
    )

def mara_mlp():
    return dict(
        num_layers = 2,
        num_hidden = 64,
        layer_norm = False,
        timesteps_per_batch = 2048,
        total_timesteps = 1e8,
        max_kl = 0.01,
        cg_iters = 10,
        gamma = 0.99,
        lam = 0.95,
        seed = 0,
        ent_coef = 0.0,
        cg_damping = 0.1,
        vf_stepsize = 1e-3,
        vf_iters = 5,
        max_episodes = 0,
        max_iters = 0,
        normalize_observations = True,
        save_interval = 10,
        env_name = 'MARA-v0',
        # env_name = 'MARAOrient-v0',
        # env_name = 'MARACollision-v0',
        # env_name = 'MARACollisionOrient-v0',
        transfer_path = None,
        # transfer_path = '/tmp/ros2learn/MARA-v0/trpo_mpi/2019-02-19_13h02min/checkpoints/best',
        trained_path = '/tmp/ros2learn/MARA-v0/trpo_mpi/2019-02-19_13h11min/checkpoints/best'
    )
