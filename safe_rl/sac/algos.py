from safe_rl.sac.sac_da_per import fsac as run_fsac
from safe_rl.sac.sac_da_per_v2 import fsac as run_fsac_v2

def sac(**kwargs):
    sac_kwargs = dict(constrained_costs=False,
                      prioritized_experience_replay=False)
    run_fsac(**sac_kwargs, **kwargs)

def sac_v2(**kwargs):
    sac_kwargs = dict(constrained_costs=False,
                      prioritized_experience_replay=False)
    run_fsac_v2(**sac_kwargs, **kwargs)

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

def fsac_v2(**kwargs):
    fsac_kwargs = dict(constrained_costs=True,
                      pointwise_multiplier=True,
                      prioritized_experience_replay=False)
    run_fsac_v2(**fsac_kwargs, **kwargs)

def fsac_per(**kwargs):
    fsac_kwargs = dict(constrained_costs=True,
                      pointwise_multiplier=True,
                      prioritized_experience_replay=True)
    run_fsac(**fsac_kwargs, **kwargs)

def fsac_per_v2(**kwargs):
    fsac_kwargs = dict(constrained_costs=True,
                      pointwise_multiplier=True,
                      prioritized_experience_replay=True)
    run_fsac_v2(**fsac_kwargs, **kwargs)

def fsac_per_dq(**kwargs):
    fsac_kwargs = dict(constrained_costs=True,
                      pointwise_multiplier=True,
                      prioritized_experience_replay=True,
                      double_qc=True)
    run_fsac(**fsac_kwargs, **kwargs)