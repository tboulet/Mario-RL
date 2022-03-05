DQN_CONFIG = {"name" : "DQN",
    "gamma" : 0.99,
    "sample_size" : 256,
    "learning_rate" : 1e-3,
        
    "reward_scaler" : None,
    "update_method" : "soft",
    "target_update_interval" : 5000,
    "tau" : 0.999,
    "double_q_learning" : True,
    "clipping" : None,
    "train_freq" : 1,
    "gradients_steps" : 1,
     
    "learning_starts" : 2000,
    "exploration_timesteps" : 10000,
    "exploration_initial" : 1,
    "exploration_final" : 0.1,
    }

REINFORCE_CONFIG = {"name" : "REINFORCE",
    "learning_rate" : 1e-4,
    "gradient_steps" : 4,   #<!> for >1, this became off policy
    "gamma" : 0.99,
    "reward_scaler" : None,
    
    "batch_size" : 1,
    
    "J_method" : "ratio_ln",   #ratio or ratio_ln     
    "epsilon_clipper" : 0.2,
    }

ACTOR_CRITIC_CONFIG = {"name" : "ACTOR_CRITIC",
    "learning_rate_actor" : 1e-2,
    "learning_rate_critic" : 1e-2,
    "compute_gain_method" : "GAE",
    "gamma" : 0.98,     
    "lmbda" : 0.98,
    "reward_scaler" : 100,
    "batch_size" : 2,        #TO IMPLEMENT. #Algorithm updates critic at every steps, and policy every batchsize steps, using the entire batch
    "gradient_steps_critic" : 4,
    "gradient_steps_policy" : 1,
    "tau" : 0.99,
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
    "learning_rate_actor" : 1e-4,
    "learning_rate_critic" : 1e-3,
    "gamma" : 0.99,
    "gae_lambda" : 0.95,
    
    "train_freq_episode" : 1,
    "n_episodes" : 4,
    "batch_size" : 128,
    "epochs" : 12,
    "n_step" : 8,
    "compute_advantage_method" : 'A_MC',    #In A_MC, A_TD, A_n_step, A_GAE
    "compute_value_method" : 'V_MC',

    "update_method" : "soft",
    "tau" : 0.99,
    "target_update_interval" : 10000,
    "reward_scaler" : 100,
    
    "epsilon_clipper" : 0.2,
    "c_critic" : 1,
    "c_entropy" : 0.01,
    }

DUMMY_CONFIG = dict()