import numpy as np
import random

class TabularActorCritic:

    def __init__(self, n_actions,
                 gamma=0.9,
                 actor_lr=0.05,
                 critic_lr=0.1,
                 temperature=1.0,
                 epsilon=0.05):

        self.n_actions = n_actions
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.temperature = temperature
        self.epsilon = epsilon

        self.prefs = {}  # Actor
        self.V = {}      # Critic

    def ensure(self, s):
        if s not in self.prefs:
            self.prefs[s] = np.zeros(self.n_actions)
        if s not in self.V:
            self.V[s] = 0.0

    def softmax(self, x):
        x = np.array(x) / max(1e-9, self.temperature)
        m = np.max(x)
        e = np.exp(x - m)
        return e / np.sum(e)

    def choose_action(self, s):
        self.ensure(s)

        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        pi = self.softmax(self.prefs[s])
        return int(np.random.choice(np.arange(self.n_actions), p=pi))

    def update(self, s, a, r, s2):

        self.ensure(s)
        self.ensure(s2)

        delta = r + self.gamma * self.V[s2] - self.V[s]

        # Critic update
        self.V[s] += self.critic_lr * delta

        # Actor update
        pi = self.softmax(self.prefs[s])

        for k in range(self.n_actions):
            if k == a:
                self.prefs[s][k] += self.actor_lr * delta * (1 - pi[k])
            else:
                self.prefs[s][k] -= self.actor_lr * delta * pi[k]