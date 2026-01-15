⏺ Now I have a complete understanding. Let me provide a comprehensive summary of how to run each method from Table 6 based on the AlphaSchedule project.

  Based on my analysis of the paper's Table 6 and the AlphaSchedule codebase, here's a comprehensive guide on how to run each of the four search methods:

  ---
  Table 6 Methods Summary

  From the paper (lines 728-746), Table 6 compares ablation results for:
  ┌─────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────┐
  │     Method      │                                           Description                                            │
  ├─────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ GPSearch        │ Guided Policy Search - uses trained network for top-K expansion + network rollout for evaluation │
  ├─────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Random Search   │ Random K-selection for expansion + random rollout for evaluation                                 │
  ├─────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Beam Search     │ All feasible actions expanded + trajectory probability for evaluation (no rollout)               │
  ├─────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Network Rollout │ Direct policy execution using argmax (no search)                                                 │
  └─────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────┘
  ---
  Step 1: Train PPO Model First

  Before running any search method, you need a trained model from ppo_policyV0.040:

  cd /Users/zhangbowen/Projects/AlphaSche/AlphaSchedule/ppo_policyV0.040

  # Train for 24 hours (adjust --run-hours as needed)
  uv run python main.py \
      --env-name ml-agent \
      --part-num 65 \
      --mach-num 30 \
      --dist-type h \
      --num-processes 8 \
      --run-hours 24 \
      --excel-save

  This saves:
  - Model checkpoint: trained_models/ppo/ml-agent-65-30-h.pt
  - Weight file: trained_models/ppo/weight.model

  ---
  Step 2: Running Each Method

  Method 1: Network Rollout (Pure Policy)

  Uses the trained network directly without any search. Located in ppo_policyV0.040/enjoy.py.

  cd /Users/zhangbowen/Projects/AlphaSche/AlphaSchedule/ppo_policyV0.040

  uv run python enjoy.py \
      --env-name ml-agent \
      --part-num 65 \
      --mach-num 30 \
      --dist-type h \
      --test-num 100 \
      --num-processes 10 \
      --load-dir ./trained_models/ppo \
      --gpu 0

  Key characteristics:
  - No search, just argmax on policy output
  - Fastest method
  - Uses deterministic=True in actor_critic.act()

  ---
  Method 2: GPSearch (Guided Policy Search)

  Uses the trained network to guide beam search with network rollout evaluation. Located in mctsAlphaV0.077/.

  Step 2a: Copy the trained weight to MCTS models directory:
  mkdir -p /Users/zhangbowen/Projects/AlphaSche/AlphaSchedule/mctsAlphaV0.077/models/
  cp /Users/zhangbowen/Projects/AlphaSche/AlphaSchedule/ppo_policyV0.040/trained_models/ppo/weight.model \
     /Users/zhangbowen/Projects/AlphaSche/AlphaSchedule/mctsAlphaV0.077/models/65-30-weight.model

  Step 2b: Update EnvirConf.py for your configuration:
  # Edit mctsAlphaV0.077/venvs/EnvirConf.py to set:
  # self.partNum = 65
  # self.distType = 'h'

  Step 2c: Run GPSearch:
  cd /Users/zhangbowen/Projects/AlphaSche/AlphaSchedule/mctsAlphaV0.077

  uv run python testPolicy.py

  Key characteristics (from mcts_policy.py:276-278):
  - mode = 'mcts_policy'
  - Top-K expansion using network probabilities
  - Rollout evaluation using network policy
  - beam_size = 10 (K parameter)

  ---
  Method 3: Beam Search (Trajectory Probability)

  Uses trajectory probability for evaluation instead of rollout. This requires the beamSearchV0.010 implementation from alpha-PPO-code.

  Option A: Copy from alpha-PPO-code:
  cp -r /Users/zhangbowen/Projects/AlphaSche/alpha-PPO-code/BeamAndRandom/beamSearchV0.010 \
        /Users/zhangbowen/Projects/AlphaSche/AlphaSchedule/beamSearchV0.010

  Option B: Run from alpha-PPO-code:
  cd /Users/zhangbowen/Projects/AlphaSche/alpha-PPO-code/BeamAndRandom/beamSearchV0.010

  # Copy model file
  mkdir -p models
  cp /path/to/trained/weight.model ./models/65-30-h-weight.model

  uv run python testPolicy.py \
      --part-num 65 \
      --mach-num 30 \
      --dist-type h \
      --beam-size 10 \
      --test-num 100

  Key difference from GPSearch (from beamSearchV0.010/mcts_policy.py:126-129):
  # Beam Search uses trajectory probability:
  grade = root._Grade_Q * node._P  # probability multiplication
  node.update(grade)

  # GPSearch uses rollout evaluation:
  grade = self._evaluate_rollout(state_copy)
  node.update(grade)

  ---
  Method 4: Random Search

  Uses random selection and random rollout (no network). Located in alpha-PPO-code/BeamAndRandom/randomRolloutV0.010/.

  cd /Users/zhangbowen/Projects/AlphaSche/alpha-PPO-code/BeamAndRandom/randomRolloutV0.010

  uv run python testPolicy.py \
      --part-num 65 \
      --mach-num 30 \
      --dist-type h \
      --beam-size 10 \
      --test-num 100

  Key characteristics (from randomRolloutV0.010/mcts_policy.py):
  - No network loaded (update_net_model = None)
  - _random_prob() for expansion (line 199-214):
  probs = np.random.rand(len(legal_actions))  # Random probabilities
  - _evaluate_random_rollout() for evaluation (line 142-160):
  max_action = np.random.choice(legal_actions)  # Random action selection

  ---
  Summary Table: Key Differences
  ┌─────────────────┬──────────────────────────────┬────────────────────────────────────┬──────────────────┐
  │     Method      │      Expansion Strategy      │        Evaluation Strategy         │ Network Required │
  ├─────────────────┼──────────────────────────────┼────────────────────────────────────┼──────────────────┤
  │ GPSearch        │ Top-K by network probability │ Network rollout (argmax)           │ Yes              │
  ├─────────────────┼──────────────────────────────┼────────────────────────────────────┼──────────────────┤
  │ Random Search   │ Random K selection           │ Random rollout                     │ No               │
  ├─────────────────┼──────────────────────────────┼────────────────────────────────────┼──────────────────┤
  │ Beam Search     │ All feasible actions         │ Trajectory probability (P product) │ Yes              │
  ├─────────────────┼──────────────────────────────┼────────────────────────────────────┼──────────────────┤
  │ Network Rollout │ N/A (direct execution)       │ N/A                                │ Yes              │
  └─────────────────┴──────────────────────────────┴────────────────────────────────────┴──────────────────┘
  ---
  Key Files Reference
  ┌─────────────────┬───────────────────────────────────┬─────────────────────────────┐
  │     Method      │             Main File             │       Mode Parameter        │
  ├─────────────────┼───────────────────────────────────┼─────────────────────────────┤
  │ Network Rollout │ ppo_policyV0.040/enjoy.py         │ N/A                         │
  ├─────────────────┼───────────────────────────────────┼─────────────────────────────┤
  │ GPSearch        │ mctsAlphaV0.077/testPolicy.py     │ policy_mode = 'mcts_policy' │
  ├─────────────────┼───────────────────────────────────┼─────────────────────────────┤
  │ Beam Search     │ beamSearchV0.010/testPolicy.py    │ policy_mode = 'mcts_policy' │
  ├─────────────────┼───────────────────────────────────┼─────────────────────────────┤
  │ Random Search   │ randomRolloutV0.010/testPolicy.py │ policy_mode = 'mcts_policy' │
  └─────────────────┴───────────────────────────────────┴─────────────────────────────┘
  ---
  Common Arguments

  --part-num     : Number of jobs (15, 25, 35, 45, 65, 95, 125)
  --mach-num     : Number of machines (auto-calculated if -1)
  --dist-type    : Distribution type ('h'=high, 'm'=medium, 'l'=low)
  --beam-size    : K parameter for search (default: 10)
  --test-num     : Number of test instances (default: 100)