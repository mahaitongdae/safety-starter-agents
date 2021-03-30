from tensorflow.python.util import deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

from safe_rl.pg.algos import ppo, ppo_lagrangian, trpo, trpo_lagrangian, cpo, ppo_dual_ascent
# from safe_rl.sac.sac import sac
from safe_rl.sac.algos import sac, sac_lagrangian, \
    sac_lagrangian_per, fsac, fsac_per, fsac_v2, sac_v2