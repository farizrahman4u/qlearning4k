import numpy as np
from random import sample


class Memory:

	def __init__(self):
		pass

	def remember(self, S, a, r, S_prime, game_over):
		pass

	def get_batch(self, model, batch_size):
		pass


class ExperienceReplay(Memory):

	def __init__(self, memory_size=100):
		self.memory = []
		self._memory_size = memory_size

	def remember(self, s, a, r, s_prime, game_over):
		self.input_shape = s.shape[1:]
		self.memory.append(np.concatenate([s.flatten(), np.array(a).flatten(), np.array(r).flatten(), s_prime.flatten(), 1 * np.array(game_over).flatten()]))
		if self.memory_size > 0 and len(self.memory) > self.memory_size:
			self.memory.pop(0)

	def get_batch(self, model, batch_size, gamma=0.9):
		if len(self.memory) < batch_size:
			batch_size = len(self.memory)
		nb_actions = model.output_shape[-1]
		samples = np.array(sample(self.memory, batch_size))
		input_dim = np.prod(self.input_shape)
		S = samples[:, 0 : input_dim]
		a = samples[:, input_dim]
		r = samples[:, input_dim + 1]
		S_prime = samples[:, input_dim + 2 : 2 * input_dim + 2]
		game_over = samples[:, 2 * input_dim + 2]
		r = r.repeat(nb_actions).reshape((batch_size, nb_actions))
		game_over = game_over.repeat(nb_actions).reshape((batch_size, nb_actions))
		S = S.reshape((batch_size, ) + self.input_shape)
		S_prime = S_prime.reshape((batch_size, ) + self.input_shape)
		X = np.concatenate([S, S_prime], axis=0)
		Y = model.predict(X)
		Qsa = np.max(Y[batch_size:], axis=1).repeat(nb_actions).reshape((batch_size, nb_actions))
		delta = np.zeros((batch_size, nb_actions))
		a = np.cast['int'](a)
		delta[np.arange(batch_size), a] = 1
		targets = (1 - delta) * Y[:batch_size] + delta * (r + gamma * (1 - game_over) * Qsa)
		return S, targets

	@property
	def memory_size(self):
		return self._memory_size

	@memory_size.setter
	def memory_size(self, value):
		if value > 0 and value < self._memory_size:
			self.memory = self.memory[:value]
		self._memory_size = value

	def reset_memory(self):
		self.memory = []
