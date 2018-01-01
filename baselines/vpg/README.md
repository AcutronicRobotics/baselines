# Vanilla Policy Gradient (VPG)

Vanilla Policy Gradient implemented using Actor-Critic method to reduce the variance.

This algorithm implements an Actor Critic Model (ACM)
which separates the policy from the value approximation process by
parameterizing the policy separately.

Pseudocode:
```
1. Initialize policy (e.g. NNs) parameter $\theta$ and baseline $b$
2. For iteration=1,2,... do
    2.1 Collect a set of trajectories by executing the current policy obtaining $\mathbf{s}_{0:H},\mathbf{a}_{0:H},r_{0:H}$
    2.2 At each timestep in each trajectory, compute
        2.2.1 the return $R_t = \sum_{t'=t}^{T-1} \gamma^{t'-t}r_{t'}$ and
        2.2.2 the advantage estimate $\hat{A_t} = R_t - b(s_t)$.
    2.3 Re-fit the baseline (recomputing the value function) by minimizing
        $|| b(s_t) - R_t||^2$, summed over all trajectories and timesteps.

          $b=\frac{\left\langle \left(  \sum\nolimits_{h=0}^{H} \mathbf{\nabla}_{\theta_{k}}\log\pi_{\mathbf{\theta}}\left(  \mathbf{a}_{h}\left\vert \mathbf{s}_{h}\right.  \right)  \right)  ^{2}\sum\nolimits_{l=0}^{H} \gamma r_{l}\right\rangle }{\left\langle \left(
          \sum\nolimits_{h=0}^{H}\mathbf{\nabla}_{\theta_{k}}\log\pi_{\mathbf{\theta}
          }\left(  \mathbf{a}_{h}\left\vert \mathbf{x}_{h}\right.  \right)  \right)
          ^{2}\right\rangle }$

    2.4 Update the policy, using a policy gradient estimate $\hat{g}$,
        which is a sum of terms $\nabla_\theta log\pi(a_t | s_t,\theta)\hat(A_t)$.
        In other words:

          $g_{k}=\left\langle \left(  \sum\nolimits_{h=0}^{H}\mathbf{\nabla
          }_{\theta_{k}}\log\pi_{\mathbf{\theta}}\left(  \mathbf{a}_{h}\left\vert
          \mathbf{s}_{h}\right.  \right)  \right)  \left(  \sum\nolimits_{l=0}^{H}
          \gamma r_{l}-b\right)  \right\rangle$
3. **end for**
```

### TODOs:
- [x] Include the pseudocode within the code
- [ ] Parameterize critic and actor networks
  - [ ] Try different network architectures
- [ ] Add more examples (besides MountainCarContinuous)
