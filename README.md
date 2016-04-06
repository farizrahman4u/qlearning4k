# qlearning4k
Q-learning for Keras

Qlearning4k is a reinforcement learning add-on for the python deep learning library [Keras](www.github.com/fchollet/keras). Its simple, and is ideal for rapid prototyping.

 **Example :**

```python
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import sgd
from qlearning4k.games import Catch
from qlearning4k import Agent

nb_frames = 1
grid_size = 10

model = Sequential()
model.add(Flatten(input_shape=(nb_frames, grid_size, grid_size)))
model.add(Dense(hidden_size, activation='relu'))
model.add(Dense(hidden_size, activation='relu'))
model.add(Dense(3))
model.compile(sgd(lr=.2), "mse")

game = Catch(grid_size)
agent = Agent(model)
agent.train(game)
agent.play()
```

