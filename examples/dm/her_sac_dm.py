import gym


import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import GoalConditionedPathCollector
from rlkit.torch.her.her import HERTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.envs.dm import DMGoalPointMassEnv

from vqvae.envs.pointmass import GoalPointmass, GoalPointmassVQVAE

from vqvae.envs.reacher import GoalReacher
from vqvae.envs.reacher import GoalReacherVQVAE, GoalReacherNoTargetVQVAE
from vqvae.envs.pusher import GoalPusher, GoalPusherNoTargetVQVAE
from vqvae.envs.utils import SimpleGoalEnv, LatentGoalEnv

MODEL_PATH = '/home/misha/research/vqvae/results/vqvae_temporal_data_long_ne8nd2.pth'
MODEL_PATH = '/home/misha/research/vqvae/results/vqvae_data_reacher_no_target_jul17_ne8nd2.pth'
MODEL_PATH = '/home/misha/research/vqvae/results/vqvae_data_pusher_no_target_jul21_ne8nd2.pth'


def experiment(variant):
    """

    eval_env = GoalPusher(threshold=0.05, reward_type='sparse',
                          max_steps=variant['algo_kwargs']['max_path_length'])
    expl_env = GoalPusher(threshold=0.05, reward_type='sparse',
                          max_steps=variant['algo_kwargs']['max_path_length'])

    eval_env = GoalReacherNoTargetVQVAE(threshold=0.2, obs_dim=128, goal_dim=128, model_path=MODEL_PATH, reward_type='sparse',
                                        max_steps=variant['algo_kwargs']['max_path_length'])

    expl_env = GoalReacherNoTargetVQVAE(threshold=0.2, obs_dim=128, goal_dim=128, model_path=MODEL_PATH, reward_type='sparse',
                                        max_steps=variant['algo_kwargs']['max_path_length'])

    eval_env = SimpleGoalEnv(obs_dim=42, goal_dim=3, env_name='stacker', reward_type='dense',
                             task='push_1', max_steps=variant['algo_kwargs']['max_path_length'])
    expl_env = SimpleGoalEnv(obs_dim=42, goal_dim=3, env_name='stacker', reward_type='dense',
                             task='push_1', max_steps=variant['algo_kwargs']['max_path_length'])
    """
    eval_env = LatentGoalEnv(obs_dim=128, goal_dim=128, reward_type='sparse',
                             threshold=0.1, max_steps=variant['algo_kwargs']['max_path_length'])
    expl_env = LatentGoalEnv(obs_dim=128, goal_dim=128, reward_type='sparse',
                             threshold=0.1, max_steps=variant['algo_kwargs']['max_path_length'])

    observation_key = 'observation'
    # ground truth goals
    desired_goal_key = 'desired_goal'
    achieved_goal_key = 'achieved_goal'
    replay_buffer = ObsDictRelabelingBuffer(
        env=eval_env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )
    obs_dim = eval_env.observation_space.spaces[observation_key].low.size
    action_dim = eval_env.action_space.low.size
    goal_dim = eval_env.observation_space.spaces[desired_goal_key].low.size

    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim + goal_dim,
        action_dim=action_dim,
        **variant['policy_kwargs']
    )
    eval_policy = MakeDeterministic(policy)
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['sac_trainer_kwargs']
    )
    trainer = HERTrainer(trainer)
    eval_path_collector = GoalConditionedPathCollector(
        eval_env,
        eval_policy,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )
    expl_path_collector = GoalConditionedPathCollector(
        expl_env,
        policy,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algo_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        algorithm='LSAC',
        version='normal',
        env_name='pusher_latent',
        title='jul25',
        save=True,
        algo_kwargs=dict(
            batch_size=128,
            num_epochs=5000,
            num_eval_steps_per_epoch=1000,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=100,
        ),
        sac_trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_rollout_goals=.2,
            fraction_goals_env_goals=0,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
    )

    def get_name(v):
        name = '_'.join([v['env_name'], v['algorithm'], v['title']])
        return name
    if variant['save']:
        name = get_name(variant)
        setup_logger(name, variant=variant)
    # optionally set the GPU (default=False)
    ptu.set_gpu_mode(True, gpu_id=0)
    experiment(variant)
