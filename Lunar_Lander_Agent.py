import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# Then we reset this environment
observation, info = env.reset()

# Create the environment
env = make_vec_env('LunarLander-v2', n_envs=16)

# Instantiate the agent
model = PPO('MlpPolicy', env, verbose=1)
# Train the agent
model.learn(total_timesteps=int(2e5))

model = PPO(
    policy = 'MlpPolicy',
    env = env,
    n_steps = 1024,
    batch_size = 64,
    n_epochs = 4,
    gamma = 0.999,
    gae_lambda = 0.98,
    ent_coef = 0.01,
    verbose=1)

# Train it for 1,000,000 timesteps
model.learn(total_timesteps=1000000)
# Save the model
model_name = "ppo-LunarLander-v2"
model.save(model_name)

# Evaluate model 
eval_env = Monitor(gym.make("LunarLander-v2"))
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

model_architecture = "PPO"

## CHANGE WITH YOUR REPO ID
repo_id = "Hasan3773/ppo-LunarLander-v2"

## Define the commit message
commit_message = "Upload PPO LunarLander-v2 trained agent"

# Create the evaluation env and set the render_mode="rgb_array"
eval_env = DummyVecEnv([lambda: gym.make(env_id, render_mode="rgb_array")])

# PLACE the package_to_hub function you've just filled here
package_to_hub(model=model, # Our trained model
               model_name=Lunar_Agent, # The name of our trained model
               model_architecture=PPO, # The model architecture we used: in our case PPO
               env_id=LunarLander-v2, # Name of the environment
               eval_env=eval_env, # Evaluation Environment
               repo_id=Hasan3773/ppo-LunarLander-v2, # id of the model repository from the Hugging Face Hub 
               commit_message='Created Lunar Agent')

env.close()
