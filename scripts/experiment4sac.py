#!/usr/bin/env python
import gym 
import safety_gym
import safe_rl
from safe_rl.utils.run_utils import setup_logger_kwargs
from safe_rl.utils.mpi_tools import mpi_fork


def main(robot, task, algo, seed, exp_name, cpu, **kwargs):

    # Verify experiment
    robot_list = ['point', 'car', 'doggo']
    task_list = ['goal1', 'goal2', 'button1', 'button2', 'push1', 'push2']
    algo_list = ['sac','sac_lagrangian','sac_lagrangian_per','fsac','fsac_per']
    algo_list.append('fsac_v2')
    algo_list.append('sac_v2')
    algo_list.append('fsac_per_v2')

    algo = algo.lower()
    task = task.capitalize()
    robot = robot.capitalize()
    assert algo in algo_list, "Invalid algo"
    assert task.lower() in task_list, "Invalid task"
    assert robot.lower() in robot_list, "Invalid robot"

    # Hyperparameters
    exp_name = algo + '_' + robot + task
    if robot=='Doggo':
        epochs = 100
        steps_per_epoch = 16e3  # max episode length: 1000
    else:
        epochs = 100
        steps_per_epoch = 16e3 # max episode length: 1000
    save_freq = 10
    cost_constraint = 9.5 # todo:3.0, add to version control

    # Fork for parallelizing
    mpi_fork(cpu)

    # Prepare Logger
    exp_name = exp_name # or (algo + '_' + robot.lower() + task.lower())
    logger_kwargs = setup_logger_kwargs(exp_name, seed)

    # Algo and Env
    algo = eval('safe_rl.'+algo)
    env_name = 'Safexp-'+robot+task+'-v0'

    algo(env_fn=lambda: gym.make(env_name),
         ac_kwargs=dict(
             hidden_sizes=(256, 256),
            ),
         epochs=epochs,
         steps_per_epoch=steps_per_epoch,
         save_freq=save_freq,
         cost_constraint=cost_constraint,
         seed=seed,
         logger_kwargs=logger_kwargs,
         **kwargs
         )

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, default='Point')
    parser.add_argument('--task', type=str, default='Button1')
    parser.add_argument('--algo', type=str, default='fsac_per_v2')
    parser.add_argument('--seed', type=int, default=0, nargs='*')
    parser.add_argument('--exp_name', type=str, default='test_lam_net')
    parser.add_argument('--cpu', type=int, default=16)
    parser.add_argument('--motivation', type=str, default='debug parallel batchsize')
    args = parser.parse_args()
    exp_name = args.exp_name if not(args.exp_name=='') else None
    main(args.robot, args.task, args.algo, args.seed, exp_name, args.cpu
         , motivation=args.motivation, version='v2')
    # main2(args.algo, args.seed, exp_name, args.cpu)
