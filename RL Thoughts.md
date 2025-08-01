Starting this notes to pen down my thoughts/questions on classic RL and it's reasoning applications in LLMs

**25/06/25**
RL is basically making a make a machine learning everything it needs to know via experience. Just as humans learn everything. Humans get feedback on their actions and thus learn whether an action is good or bad. Same way we train machine giving it feedback (positive and negative). 

But I read somewhere that positive feedback aren't as effective as negative ones. I wonder why? 2-3 reasons for it. 
- Negative feedback helps reduce the policy space, it gives clear notion to don't repeat the mistake
- Positive feedback is rare in sparse reward environments. It cannot give as much positive signals to improve learning. In addition to that, positive feedback merely says, this is good action - you "may" repeat this. There could be other actions far better than this one so the action space is not reduced.
==**Reward shaping**== is a thing in Modern RL - even for reasoning LLMs - further study

DQN is more into value functions and I have created a Atari agent which plays PONG. So far I have been able to train it so that my agent got a score of 10 but lost the match. One peculiar thing I noticed is that the Adam optimizer decay rate is very crucial. I am not very sure if decaying it every 5 episodes was a better option than decaying after 10 episodes. Ideally if it decays after 5 episodes, it should use learned strategies more than exploring stuff, which means it should give better rewards - unless the previous strategies were majorly bad. However, with 10-episode decay I got somewhat better results. I'll probably try it out with UCB instead of Adam optimizer.

I am starting with Policy gradients and optimization. Policy gradients look very intuitive since it directly trains on what actions are good at this particular moment, rather than finding the value function assigned to it and then selecting. Even ==**Likelihood Ratios**== are quite intuitive :
- How much more or less likely is action ***X***(tau) in new policy vs the old policy
$$ π_{θ_{new}​}​(τ)/π_{θ_{old​}}​(τ)​ $$
- Policy gradient RL - estimated gradient of expected return
$$ ∇_θ​J(θ) = E_{τ∼π_θ}​​[∇_θ​logπ_θ​(τ)R(τ)] $$


**==Gradient ascent==**
$$θ←θ+α∇_θ​J(θ)$$
- What does gradient signify?
	It signifies the direction and magnitude of **steepest ascent**.
- Why we need the gradient (or slope) for this ascent?
	Because it signifies the direction of steepest ascent. Not just the direction we require the magnitude of this ascent because when it multiplies with the learning rate(α) it gives the magnitude of step to be take.
If the gradient is small -> signifies that the function is rather flat, and we need to take smaller steps to avoid overshooting.
If the gradient is large -> signifies the function is changing rapidly, and we can take larger steps

**18/07/25**
**==GAE - lambda==** (Generalized Advantage Estimator)
Advantage function calculates how much we gain by taking an action a instead from state s, instead of staying at state s.
$$A(s_t,a_t) = Q(s_t,a_t) - V(s_t)$$
Calculating the advantage of an action during policy gradient RL, we have 2 types of estimator :
1. **Monte-Carlo Estimation (High Variance, Low Bias)**
	It take the actual return at the end of the episode (sum of future discounted rewards)$$A(s_t,a_t) = G_t - V(s_t)$$
	Low Bias - since it is actual return
	High Variance - Rewards will be noisy and sparse. In addition to this, the final reward cannot be used to judge each action , some intermediate action can be good but later on due to some other bad action the final reward was bad. Hence this adds variance.
2. **Temporal Difference (TD) Estimation (Low Variance, High Bias)**
	The return is calculated after each step. One-step TD error to estimate advantage also called ==TD-residual==.$$A(s_t,a_t)=r_t + γV(s_{t+1}) - V(s_t)$$
	High Bias - it relies on next estimated value function $V(s_{t+1})$ which is often inaccurate leading to high bias
	Low Variance - it takes immediate rewards and next estimated value, making it less noisy

GAE- λ takes the best of both worlds.
$$A_t^{GAE}​=∑_{l=0}^∞​(γλ)^lδ_{t+l}$$
If $λ = 0$  -> then advantage $A_t = δ_t$ which is one-step TD error having high bias, low variance
If $λ = 1$ -> then advantage $A_t$ becomes full discounted sum of all future rewards which is equivalent to $A_t = R_t - V(s_t)$ (rewards to go minus the baseline value - monte carlo estimation) having low bias, high variance
Hence we always keep the $λ$ value to be between 0 to 1 (for eg. - 0.97) to maintain the low bias-variance giving GAE the correct blend of all possible lookaheads.

**31/07/25**
**==REINFORCE==**
**Intuition:**
increase the probability of good outcomes and decrease the probability of bad outcomes by reinforcing good actions via policy

**Algorithm:**
1. Initialize a policy, which gives out the probability of each action $a$ given a state $s$. The policy network has weights as parameters $θ$.
2. Collect trajectory - Run a complete episode from start to end and record all the states, actions, rewards. This sequence is called trajectory - $s_0, a_0, r_0, s_1, a_1, r_1,... ,r_T$
3. Calculate rewards-to-go - For each action in the sequence, calculate sum of all rewards (rewards-to-go) starting from that index to the end of episode. This shows how good the outcome was following the action $a_T$. $$R_t​=∑_{k=t}^T​r_k​$$
4. Calculate loss - we take negative of the summation of rewards as the loss to minimize. Hence if we minimize the neg of rewards, we actually increase the overall rewards. Here we can see that if at time $t$ we get a very good reward to go $R_t$ then the log prob of that particular action increase as it multiplies with the $R_t$. At the same time, if the $R_t$ is not very good then the log prob of that particular action reduces thereby ensuring that that particular action occurs less often from next time onwards. $$Loss=−∑_{t=0}^T​R_t​⋅log_{π_θ}​(a_t​∣s_t​)$$
5. Update the policy - use optimizer like Adam to tweak the weights based on minimizing the loss.

**01/08/25**
**==Actor-Critic==**
**Intuition:**
REINFORCE uses entire rewards-to-go to judge each action, which means a single lucky/unlucky event late in the episode can drastically change the rewards of all prior actions making the training unstable and slow. We need immediate feedback for each action taken. So we have 2 networks - actor and critic. Actor takes the action similar to REINFORCE and Critic gives immediate feedback on the action taken (on the basis of trajectories of prior experience)

**Algorithm**
1. Initialize two neural networks - Actor (a policy network $π_θ$) and Critic (a value function $V_ϕ​$)
2. Act and collect - Given a state $s$, Actor picks an action $a$ according to policy $π_θ$. It receives reward $r$ and next state $s'$  
3. Calculate advantage using Critic model - Instead of waiting for whole episode to get complete, Critic gives immediate feedback. We calculate TD (Temporal Difference) error ($δ$) which gives us low variance advantage estimate. $$δ=r+γV_ϕ​(s')−V_ϕ​(s)$$
   Here $r$ is the reward received after acting on state $s$. The Critic model gives the value for state $s$ and $s'$. The new updated value of current state is given by $r+γV_ϕ​(s')$ and $δ$ represents "surprise" i.e. how much better/worse the outcome was than the Critic predicted.
4. Update the Actor - The loss is calculated as follows for Actor model : $$Loss=-δ*log_{π_θ}(a_t|s_t)$$
   Here there could be 2 cases:
   (a) $δ$ is positive - which means the new outcome is much better (better surprise) than the previous, hence we increase the probability of $a$ in state $s$. 
   (b) $δ$ is negative - which means the new outcome is worse (bad surprise) than the previous, hence we decrease the probability of $a$ in state $s$.
5. Update the Critic - We need to update the Critic to improve its ability to evaluate states. The loss is calculated as the square of TD error $δ^2$ and we train it to estimate $V(s)$ to be closer to target $r+γVϕ​(s′)$ 

**==Actor-Critic with GAE==**
**Intuition:**
TD error reduces the variance but introduces high bias, since it only sees the next state. There can be a case where a particular action might seem bad immediately but was a good action for the entire episode. And vice versa. Hence to reduce this bias we use Generalized Advantage Estimation (GAE) where we take TD errors from several steps to get the bias-variance balance.

**Algorithm:**
1. Initialize 2 neural nets - Actor and Critic
2. Collect the trajectories - Run the policy for fixed number of steps(a "path" or "rollout") and store the states, actions and rewards.
3. Calculate the TD error - For that trajectory calculate the TD error for each step $$δ_t=r_t+γV_ϕ​(s_{t+1})−V_ϕ​(s_t)$$
4. Calculate the GAE advantage - GAE balances the bias-variance using discounted sum of TD errors collected: $$A_t^{GAE}​=∑_{l=0}^{T−t}​(γλ)^lδ_{t+l}​$$
   The lambda $λ$ controls the bias-variance trade-off and is between 0 and 1.
5. Update the Actor - Using the GAE, we calculate the loss for actor as the average over entire trajectory :  $$−1/T​∑_{t=0}^T​A_t^{GAE}​⋅logπ_θ​(a_t​∣s_t​)$$
6. Update the Critic - We want to improve the prediction of value of state. A good target for this is actual discounted reward-to-go ($R_t$) from trajectory.
   Calculate the reward-to-go for each step : $$R_t​=∑_{k=t}^T​γ^{k−t}r_k​.$$
   Minimize the loss i.e minimize the difference between actual rewards and predicted value functions: $$1/T∑_{t=0}^T​(V_ϕ​(s_t​)−R_t​)^2$$
7. Repeat - Discard old trajectory and repeat from step 2.