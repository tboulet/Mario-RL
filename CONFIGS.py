DQN_CONFIG = {"name" : "DQN",
    "gamma" : 0.99,
    "sample_size" : 32,
    
    "frames_skipped" : 5,
    "history_lenght" : 1,   #To implement
    
    "reward_scaler" : None,
    "update_method" : "periodic",
    "target_update_interval" : 5000,
    "tau" : 0.96,
    "double_q_learning" : False,
    "clipping" : None,
    "train_freq" : 1,
    "gradients_steps" : 1,
     
    "learning_starts" : 1,
    "exploration_timesteps" : 1,
    "exploration_initial" : 0.1,
    "exploration_final" : 0.1,
    }

REINFORCE_CONFIG = {"name" : "REINFORCE",
    "learning_rate" : 1e-3,
    "gamma" : 0.99,
    "reward_scaler" : None,
    "batch_size" : 1, #TO IMPLEMENT
    "off_policy" : False,
    }

ACTOR_CRITIC_CONFIG = {"name" : "ACTOR_CRITIC",
    "learning_rate_actor" : 1e-4,
    "learning_rate_critic" : 1e-3,
    "compute_gain_method" : "total_future_reward_minus_state_value",
    "gamma" : 0.99,     
    "reward_scaler" : 100,
    "batch_size" : 1,        #TO IMPLEMENT. #Algorithm updates critic at every steps, and policy every batchsize steps, using the entire batch
    "gradient_steps_critic" : 1,
    "gradient_steps_policy" : 1,
    "clipping" : None,
    }

DDPG_CONFIG = {"name" : "DDPG",
    "learning_rate_actor" : 1e-4,
    "learning_rate_critic" : 1e-3,
    "gamma" : 0.99,     
    "reward_scaler" : 100,
    "sample_size" : 32,    
    "gradient_steps" : 1,
    "clipping" : None,
    }

PPO_CONFIG = {"name" : "PPO",
    "learning_rate" : 1e-3,
    "gamma" : 0.99,
    "timesteps" : 512,
    "batch_size" : 64,
    "epochs" : 4,
    "tau" : 0.99,
    
    "update_method" : "soft",
    "reward_scaler" : None,
    
    "epsilon_clipper" : 0.2,
    "c_critic" : 1.,
    "c_entropy" : 0.01,
    "entropy" : False,
    }

DUMMY_CONFIG = dict()