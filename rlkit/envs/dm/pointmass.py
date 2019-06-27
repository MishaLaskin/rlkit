from dm_control import suite
import numpy as np
from gym.spaces import Box, Dict


class DMPointMassEnv:
        
        def __init__(self, 
                     env_name='point_mass',
                     reward_type='dense',
                     indicator_threshold=0.06,
                     mode='easy',
                     max_steps=100
                    ):
            
            self.env_name = env_name 
            self.mode = mode
            self.dm_env = suite.load(self.env_name,self.mode)
            
            self.max_steps = max_steps
            self.indicator_threshold = indicator_threshold
            self.reward_type = reward_type
            self.action_spec = self.dm_env.action_spec()
            self.obs_spec = self.dm_env.observation_spec()
            self.action_space = Box(self.action_spec.minimum, self.action_spec.maximum, dtype=np.float32)
            self.observation_space = Box(low=-5.0,high=5.0,shape=(np.sum([x.shape for x in self.obs_spec.values()]),), dtype=np.float32)
            
        def _distance_to_target(self):
            return self.dm_env.physics.mass_to_target_dist()
        
        def _time_step_to_obs(self,ts):
            state = np.concatenate([s for s in ts.observation.values()])
            return state
        
        def _time_step_to_s_r_d_info(self,action,ts):
            state = np.concatenate([s for s in ts.observation.values()])
            reward = self.compute_reward(action,state)
            done = True if self.env_step >= self.max_steps else False
            info = {}
            info['dm_reward'] = ts.reward
            info['discount'] = ts.discount
            info['step_type'] = ts.step_type
            return state, reward, done, info
        
        def compute_reward(self, action, obs):
            
            if self.reward_type == 'dense':
                distance_to_target = self._distance_to_target()
                return - distance_to_target
            else:
                raise NotImplementedError('Reward type not specified')
            
        def reset(self):
            self.env_step = 0
            time_step = self.dm_env.reset()
            state = self._time_step_to_obs(time_step)
            return state
        
        def step(self,action):
            assert action.shape == self.action_spec.shape, 'Action must have shape '+str(action_spec.shape)
            
            self.env_step +=1
            time_step = self.dm_env.step(action)
            return self._time_step_to_s_r_d_info(action,time_step)
        
        def render(self,w=128,h=128,camera_id=0):
            return self.dm_env.physics.render(w, h, camera_id=camera_id)
        
        def _generate_goal_img(self,w=128,h=128,camera_id=0,debug=False):
            env = suite.load(self.env_name,self.mode)
            
            env.reset()
            while env.physics.mass_to_target_dist() > .1:
                env.reset()
                start_img = env.physics.render(w, h, camera_id=camera_id)
                
            while env.physics.mass_to_target_dist() > self.indicator_threshold:
                a = self.action_space.sample()
                env.step(a)
                
            goal_img = env.physics.render(w, h, camera_id=camera_id)
            if debug:
                return start_img, goal_img
            else:
                return goal_img
                
        
if __name__ == "__main__":
    env = DMPointMassEnv()
    obs = env.reset()
    print('reset obs',obs)
    obs_,r,d,info = env.step(env.action_space.sample())
    print('next obs',obs_)
    print('reward',r)
    print('distance to target',env._distance_to_target())
    print('img shape',env.render().shape)


