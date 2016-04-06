import numpy as np
from random import sample

class Memory:

	def __init__(self):
		pass

	def remember(self, transition, game_over):
		pass

	def get_batch(self, model, batch_size):
		pass


class ExperienceReplay(Memory):

	def __init__(self, memory_size=100):
		self.memory = []
		self._memory_size = memory_size

	def remember(self, transition, game_over):
		# transition = [state, action, reward, new_state]
		self.memory.append([transition, game_over])
		if self.memory_size > 0 and len(self.memory) > self.memory_size:
			self.memory.pop(0)

	def get_batch(self, model, batch_size, discount=0.9):
		if len(self.memory) < batch_size:
			batch_size = len(self.memory)
		# Pick 'batch_size' number of random elements from memory
		batch = sample(self.memory, batch_size)
		S = np.array(map(lambda x:x[0][0], batch))  # (batch_size, 1, nb_frames) + game.output_shape
		S = np.squeeze(S, 1)  # (batch_size, nb_frames) + game.output_shape
		S_prime = np.array(map(lambda x:x[0][3], batch))  # (batch_size, 1, nb_frames) + game.output_shape
		S_prime = np.squeeze(S_prime, 1)  # (batch_size, nb_frames) + game.output_shape
		X = np.concatenate([S, S_prime], axis=0)  # (2 * batch_size, nb_frames) + game.output_shape
		Y = model.predict(X)  # (2 * batch_size, nb_actions)
		a = Y[:batch_size]  # (batch_size, nb_actions)
		r = np.max(Y[batch_size:], axis=1)  # (batch_size,)`
		# Set correct reward
		for i, experience in enumerate(batch):
			transition = experience[0]
			action = transition[1]
			reward = transition[2]
			game_over = experience[1]
			if not game_over:
				reward += discount * r[i]
			a[i, action] = reward
		return S, a

	@property
	def memory_size(self):
	    return self._memory_size

	@memory_size.setter
	def memory_size(self, value):
		if value > 0 and value < self._memory_size:
			self.memory = self.memory[:value]
		self._memory_size = value
