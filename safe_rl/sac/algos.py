from safe_rl.sac.sac_da_per import fsac as run_fsac

def sac(**kwargs):
    sac_kwargs = dict(constrained_costs=False,
                      prioritized_experience_replay=False)
    run_fsac(**sac_kwargs, **kwargs)

def sac_lagrangian(**kwargs):
    sac_kwargs = dict(constrained_costs=True,
                      pointwise_multiplier=False,
                      prioritized_experience_replay=False)
    run_fsac(**sac_kwargs, **kwargs)

def sac_lagrangian_per(**kwargs):
    sac_kwargs = dict(constrained_costs=True,
                      pointwise_multiplier=False,
                      prioritized_experience_replay=True)
    run_fsac(**sac_kwargs, **kwargs)

def fsac(**kwargs):
    fsac_kwargs = dict(constrained_costs=True,
                      pointwise_multiplier=True,
                      prioritized_experience_replay=False)
    run_fsac(**fsac_kwargs, **kwargs)

def fsac_per(**kwargs):
    fsac_kwargs = dict(constrained_costs=True,
                      pointwise_multiplier=True,
                      prioritized_experience_replay=True)
    run_fsac(**fsac_kwargs, **kwargs)
