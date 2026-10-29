"""Syllabus experiment: compare epsilon-greedy and UCB on a simulated bandit.
This is a toy simulation for algorithm understanding, not evidence of real-world care suitability.
Run: python experiments/bandit_simulation.py
"""
import numpy as np

SEED = 17
ARMS = 5
ROUNDS = 1500
TRUE_MEANS = np.array([0.18, 0.42, 0.31, 0.56, 0.38])

def simulate_epsilon_greedy(rng, epsilon=0.1):
    counts=np.zeros(ARMS); values=np.zeros(ARMS); rewards=[]; regret=[]
    for t in range(ROUNDS):
        if rng.random()<epsilon: arm=int(rng.integers(ARMS))
        else: arm=int(np.argmax(values))
        reward=float(rng.random()<TRUE_MEANS[arm]); counts[arm]+=1
        values[arm]+=(reward-values[arm])/counts[arm]
        rewards.append(reward); regret.append(float(TRUE_MEANS.max()-TRUE_MEANS[arm]))
    return np.array(rewards),np.array(regret),counts

def simulate_ucb(rng, c=1.5):
    counts=np.zeros(ARMS); values=np.zeros(ARMS); rewards=[]; regret=[]
    for t in range(ROUNDS):
        untried=np.flatnonzero(counts==0)
        if len(untried): arm=int(untried[0])
        else: arm=int(np.argmax(values+c*np.sqrt(np.log(t+1)/counts)))
        reward=float(rng.random()<TRUE_MEANS[arm]); counts[arm]+=1
        values[arm]+=(reward-values[arm])/counts[arm]
        rewards.append(reward); regret.append(float(TRUE_MEANS.max()-TRUE_MEANS[arm]))
    return np.array(rewards),np.array(regret),counts

if __name__=='__main__':
    for name, fn in [('epsilon-greedy',simulate_epsilon_greedy),('UCB',simulate_ucb)]:
        r,g,c=fn(np.random.default_rng(SEED))
        print(f'{name:16s} | cumulative reward={r.sum():.0f}/{ROUNDS} | average reward={r.mean():.3f} | cumulative regret={g.sum():.2f} | arm pulls={c.astype(int).tolist()}')
    print('\nSynthetic benchmark only. Replace simulated rewards with judged offline feedback before drawing conclusions about real recommendations.')
