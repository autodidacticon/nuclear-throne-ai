To train a grandmaster-level artificial agent in **Nuclear Throne** (NT)—a fast-paced, procedurally generated, bullet-hell roguelike—we must adapt DeepMind’s **AlphaStar** methodology from a PvP Real-Time Strategy context into a PvE environment. 

Because you have full control over the engine via the `nt-recreated-public` repository, you can bypass the extreme compute inefficiencies of pixel-based Computer Vision (RL from pixels) and interact directly with the game's internal memory.

Here is a high-level Machine Learning architectural guide and training plan.

---

### 1. Environment Interface (Formulating the POMDP)
Nuclear Throne is a Partially Observable Markov Decision Process (POMDP). The agent must infer off-screen threats and map layouts. AlphaStar succeeded by using a highly structured "raw interface"; we will do the same.

**Observation Space (State Inputs):**
*   **Dynamic Entity List:** A variable-length array of active entities (Enemies, Projectiles, Rads, Weapons, Chests). Each entity is a feature vector: `[Type_ID, Rel_X, Rel_Y, Vel_X, Vel_Y, HP, State_Flags]`. *Projectiles are the most critical entities here.*
*   **Spatial Feature Grid (Minimap):** A localized 2D grid (e.g., $64 \times 64$ tiles) centered on the player to represent static geometry (Walkable Floor, Walls, Unbreakable Walls, Hazards, Portals).
*   **Scalar Context:** Global player state parameters: `[Current HP, Max HP, Ammo Counts, Current Weapons, Current Level/Loop, Active Mutations]`.

**Action Space (Autoregressive Policy):**
NT requires simultaneous dodging, aiming, and resource management. To manage the combinatorial explosion of concurrent actions, structure the output autoregressively (where the network predicts Action A, then uses A to condition the prediction of Action B in the same frame):
1.  **Movement:** Discretized 8-way directional pad (or continuous vector).
2.  **Aiming:** Continuous 360-degree angle (conditioned on Movement).
3.  **Discrete Actions (Binary):** `Primary Fire`, `Active Ability`, `Swap Weapon`, `Interact`.
*Note: To ensure the agent learns strategy rather than relying entirely on robotic, frame-perfect micro-dodging, enforce a ~150ms artificial reaction delay to mirror human limitations, similar to AlphaStar’s APM constraints.*

---

### 2. Neural Network Architecture (The "Torso")
The network must process an arbitrary number of bullets, understand spatial cover, and maintain temporal memory.

*   **Entity Encoder (Transformer / Self-Attention):** Feed the Dynamic Entity List into a Transformer network. Self-attention allows the agent to dynamically prioritize threats (e.g., heavily weighting a high-velocity sniper bullet over a slow-moving maggot) regardless of their order in the array.
*   **Spatial Encoder (ResNet):** Pass the 2D Spatial Grid through a Convolutional Residual Network to extract geometric features like chokepoints, dead ends, and line-of-sight.
*   **Core Memory (Deep LSTM):** Concatenate the outputs of the Transformer, ResNet, and Scalar Context into a Deep Long Short-Term Memory (LSTM) network. The LSTM is critical for tracking off-screen enemies, anticipating weapon reload timings, and remembering dropped ammo chests.
*   **Action & Value Heads:** The LSTM feeds into sequential policy heads (Actor) to output the autoregressive actions, and a Value head (Critic) to estimate expected future rewards.

---

### 3. Phase 1: Bootstrapping via Imitation Learning
A randomly initialized RL agent dropped into a bullet-hell will die instantly, generating zero positive reward signal. AlphaStar solved the exploration problem by bootstrapping the network with human data.

1.  **Data Collection:** Use the source code to log state-action pairs silently during human gameplay at 30Hz. Collect ~50 hours of high-level human runs.
2.  **Behavioral Cloning:** Train the architecture via Supervised Learning to predict the human’s actions given the state (using Cross-Entropy for discrete actions, MSE for continuous).
3.  **The Baseline Policy ($\pi_{SL}$):** The model learns basic survival heuristics—dodging glowing red entities, pointing the crosshair at enemies, and pathfinding to the portal.

---

### 4. Phase 2: Distributed Reinforcement Learning
Once the agent mimics human competence, transition to Reinforcement Learning to optimize for superhuman reflexes.

1.  **Algorithm (IMPALA with V-Trace):** Utilize an asynchronous, off-policy Actor-Critic architecture. Hundreds of headless game instances (Actors) run in parallel at highly accelerated tick rates, sending trajectories to a centralized GPU Learner. **V-Trace** corrects the policy lag between the actors generating the data and the learner updating the weights.
2.  **UPGO (Upgoing Policy Update):** A core AlphaStar technique. UPGO forcefully updates the policy network from trajectories that exceeded the Value network's expectations. If the agent pulls off a miraculous dodge to survive a boss fight, UPGO ensures this high-value micro-behavior is rapidly reinforced.
3.  **KL-Divergence Regularization:** To prevent catastrophic forgetting, apply a Kullback-Leibler (KL) divergence penalty between the RL agent and the baseline Supervised human model ($\pi_{SL}$). This ensures the agent doesn't forget basic mechanics while randomly exploring.
4.  **Reward Shaping:**
    *   *Dense Rewards (Early RL):* +0.1 per Rad, +1 per Kill, -5 for taking damage.
    *   *Sparse Rewards (Late RL):* Decay dense rewards to rely on sparse signals: +50 for Boss Kills, +500 for Looping. This prevents the agent from "reward hacking" (e.g., farming weak enemies infinitely instead of progressing).

---

### 5. Phase 3: Adapting "League Training" for PvE
AlphaStar used a League of agents playing against each other to prevent exploiting single strategies (Nash Equilibrium). For a PvE roguelike, the agent risks **policy collapse** by overfitting to one playstyle (e.g., mastering hitscan weapons but dying instantly when forced to use explosives). We adapt this into a **Population-Based Curriculum**:

1.  **Population-Based Training (PBT):** Maintain a population of agents. Randomly mutate their hyperparameters (learning rates, reward weights) over time. Copy the hyperparameters of the agents that survive the longest, organically evolving the optimal training setup.
2.  **Constraint Specialists (Exploiters):** Train parallel branches of the agent forced to play under strict conditions using your source-code manipulation:
    *   *Melee Specialist:* Only allowed to use melee weapons. Learns frame-perfect bullet deflection and aggressive spacing.
    *   *Explosive Specialist:* Forced to use Bazookas. Learns blast-radius spacing and avoids self-damage.
    *   *Fragile Specialist:* Forced to play *Melting* (2 Max HP). Learns flawless no-hit evasion.
3.  **Policy Distillation:** Periodically, distill the network weights of these highly-tuned Specialists back into the Main Generalist Agent, granting it a robust repertoire of micro-mechanics.

---

### 6. Phase 4: Long-Term Credit Assignment (Mutations)
Choosing level-up Mutations (e.g., *Rhino Skin* vs. *Plutonium Hunger*) is a discrete, low-frequency, high-impact meta-decision. Standard RL struggles to connect a choice made on Level 1 to a death on Level 7.

*   **Hierarchical RL Manager:** Treat the Mutation selection screen as a separate sub-policy. When the level-up screen appears, pause the main reflex-action loop. 
*   **Value-Estimation Bandit:** Pass the agent's current loadout, character, and the available mutation choices into a secondary Value Network. This network evaluates which mutation choice maximizes the *long-term probability of Looping*, bypassing the short-term reflex reward system entirely.