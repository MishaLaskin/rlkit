from dm_control import suite
import numpy as np
from gym.spaces import Box, Dict
import os
import torch
from vqvae.models.vqvae import VQVAE
from rlkit.envs.dm.point_mass import DMImageGoalPointMassEnv, DMPointMassEnv
from vqvae.planner import RepresentationGraph
import numpy as np
from vqvae import utils


class DMImageGoalPointMassEnvWithVQVAE(DMImageGoalPointMassEnv):

    def __init__(self,
                 graph_file=None,
                 model_filename=None,
                 model_dir=None,
                 min_rep_count=100,
                 experiment='her',
                 **kwargs):

        DMImageGoalPointMassEnv.__init__(self, **kwargs)

        if model_filename is None:
            raise ValueError(
                'Must supply a valid path to VQVAE model .pth file')
        self.experiment = experiment
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.model_params = self._load_model(
            model_filename, model_dir)

        if graph_file is None:
            rep_graph = RepresentationGraph(model_filename=model_filename,
                                            model_dir=model_dir,
                                            min_rep_count=100)

            self.graph = rep_graph.graph
            self.rep_dict = rep_graph.rep_dict
        else:
            rep_graph = np.load(graph_file)
            self.graph = rep_graph.item().get('graph')
            self.rep_dict = rep_graph.item().get('rep_dict')
            self.rep_to_state = rep_graph.item().get('rep_to_state')

        self.reps_state_space = Box(
            low=-1.0, high=1.0, shape=(64,), dtype=np.float32)
        self.image_space = Box(low=0, high=255, shape=(
            self.img_dim, self.img_dim, self.num_channels))

        self.observation_space = Dict([
            ('observation', self.reps_state_space),
            ('desired_goal', self.reps_state_space),
            ('achieved_goal', self.reps_state_space),
            ('state_observation', self.reps_state_space),
            ('state_desired_goal', self.reps_state_space),
            ('state_achieved_goal', self.reps_state_space),
            ('image_desired_goal', self.image_space),
            ('image_achieved_goal', self.image_space),
        ])

    def _load_model(self, model_filename, model_dir=None):
        # if running from the VQ VAE folder, you can set model_dir = None
        path = os.getcwd() + '/results/' if model_dir is None else model_dir

        if torch.cuda.is_available():
            data = torch.load(path + model_filename)
        else:
            data = torch.load(path+model_filename,
                              map_location=lambda storage, loc: storage)

        params = data["hyperparameters"]

        model = VQVAE(params['n_hiddens'], params['n_residual_hiddens'],
                      params['n_residual_layers'], params['n_embeddings'],
                      params['embedding_dim'], params['beta']).to(self.device)

        model.load_state_dict(data['model'])

        return model, params
    """
    def _obs_to_goal_obs(self, obs, debug=False):

        achieved_goal, desired_goal = self._get_pointmass_and_target_pos()
        achieved_goal_image = self._get_achieved_goal_img()

        # encode into a representation hash
        achieved_rep_state = self._encode_image(achieved_goal_image)
        #

        goal_obs = {
            "observation": obs,
            "desired_goal": desired_goal,
            "achieved_goal": achieved_goal,
            "state_observation": obs,
            "state_desired_goal": self.desired_rep_state,
            "state_achieved_goal": achieved_rep_state,
            "image_desired_goal": self.desired_goal_image,
            "image_achieved_goal": achieved_goal_image,
        }

        return goal_obs
    """

    def reset(self):

        if self.experiment == 'her':
            goal_obs = DMImageGoalPointMassEnv.reset(self)
            achieved_rep = self._encode_image(goal_obs["image_achieved_goal"])
            self.desired_rep = self._encode_image(
                goal_obs["image_desired_goal"])
            self.desired_rep = (self.desired_rep - 8.0)/8.0
            achieved_rep = (achieved_rep-8.0)/8.0
            goal_obs['state_desired_goal'] = self.desired_rep
            goal_obs['state_achieved_goal'] = achieved_rep
            goal_obs['state_observation'] = achieved_rep
            goal_obs['desired_goal'] = self.desired_rep
            goal_obs['achieved_goal'] = achieved_rep
            goal_obs['observation'] = achieved_rep
            self.current_rep = achieved_rep
            # resets again to overwrite cur

        else:
            obs = self._reset_to_state_from_graph()
            # resets again to overwrite current state (which is the goal state)
            goal_obs = self._obs_to_goal_obs(obs)
        return goal_obs

    def compute_reward(self, action, obs, *args, **kwargs):
        if np.linalg.norm(self.current_rep - self.desired_rep, 2) == 0:
            return 0.
        else:
            return -1.

    def step(self, action, debug=False):
        if debug:
            start = time.time()

        obs_next, reward, done, info = DMPointMassEnv.step(self, action)
        goal_obs_next = DMImageGoalPointMassEnv._obs_to_goal_obs(
            self, obs_next)
        achieved_rep = self._encode_image(goal_obs_next["image_achieved_goal"])
        achieved_rep = (achieved_rep-8.0)/8.0
        goal_obs_next['state_desired_goal'] = self.desired_rep
        goal_obs_next['state_achieved_goal'] = achieved_rep
        goal_obs_next['state_observation'] = achieved_rep
        goal_obs_next['desired_goal'] = self.desired_rep
        goal_obs_next['achieved_goal'] = achieved_rep
        goal_obs_next['observation'] = achieved_rep
        self.current_rep = achieved_rep

        if debug:
            print('Time per step', time.time()-start)
            print('Size of obs obj', getsizeof(goal_obs_next))

        return goal_obs_next, reward, done, info

    def _reset_her_experiment(self):
        goal_obs = DMImageGoalPointMassEnv.reset(self)

    def _reset_to_state_from_graph(self):
        obs = DMPointMassEnv.reset(self)

        def get_rand(d):
            return np.random.choice(list(d.keys()))

        start_rep, end_rep = get_rand(
            self.rep_dict), get_rand(self.rep_dict)

        self.desired_rep_state = end_rep

        goal_state = self.rep_to_state[end_rep]
        start_state = self.rep_to_state[start_rep]

        # set simulator to goal state
        # get image obs

        self.dm_env.physics.named.data.geom_xpos['pointmass'] = goal_state
        goal_img = self.render(self.img_dim, self.img_dim)

        # set simulator to start state
        # get image obs

        self.dm_env.physics.named.data.geom_xpos['pointmass'] = start_state

        return goal_img

    def _encode_image(self, img):
        img = np.array([img]).transpose(0, 3, 1, 2)
        img = torch.tensor(img).float()
        img = img.to(self.device)
        vq_encoder_output = self.model.pre_quantization_conv(
            self.model.encoder(img))
        _, _, _, _, e_indices = self.model.vector_quantization(
            vq_encoder_output)
        return e_indices.cpu().detach().numpy().squeeze()


if __name__ == "__main__":
    #model_dir = '/home/misha/downloads/vqvae/results/'
    #model_filename = 'vqvae_data_point_mass_v2ne16nd16.pth'
    #graph_file = model_dir + 'vqvae_graph_point_mass_v2ne16nd16.npy'
    model_dir = '/home/misha/downloads/vqvae/results/'
    model_filename = 'vqvae_data_point_mass_jul2_ne16nd16.pth'
    graph_file = '/home/misha/downloads/vqvae/results/small_graph.npy'

    print('loading graph and model')
    env = DMImageGoalPointMassEnvWithVQVAE(graph_file=graph_file,
                                           model_dir=model_dir,
                                           model_filename=model_filename)
    obs = env.reset()
    print('reset obs', obs.keys())
    obs_, r, d, info = env.step(env.action_space.sample())
    print('next obs', obs_.keys())
    print('reward', r)
    print('distance to target', env._distance_to_target())
    print('img shape', env.render(32, 32).shape)

    print('achieved', (obs_['state_achieved_goal']-8.)/16.)

    print('desired', obs_['state_desired_goal'])
