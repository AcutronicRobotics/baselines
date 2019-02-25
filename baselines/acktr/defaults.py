def mujoco():
    return dict(
        nsteps=2500,
        value_network='copy'
    )

def mara():
    return dict(
        num_layers=2,
        num_hidden=64,
        layer_norm=False,
        nsteps = 1024,
        nprocs=8,
        gamma = 0.99,
        lam = 0.97,
        ent_coef=0.01,
        vf_coef=0.5,
        vf_fisher_coef=1.0,
        lr=0.25,
        max_grad_norm=0.5,
        kfac_clip=0.001,
        is_async=True,
        seed = 0,
        total_timesteps = int(1e8),
        # network = 'cnn',
        value_network = 'copy',
        lrschedule='linear',
        log_interval=1,
        save_interval = 10,
        # env_name = 'MARA-v0',
        # env_name = 'MARAOrient-v0',
        env_name = 'MARACollision-v0',
        # env_name = 'MARACollisionOrient-v0',
        transfer_path = None,
        # transfer_path = '/tmp/ros_rl2/MARACollision-v0/acktr/2019-02-25_13h53min/checkpoints/best,
        trained_path = '/tmp/ros_rl2/MARACollision-v0/acktr/2019-02-25_13h53min/checkpoints/best'
    )
