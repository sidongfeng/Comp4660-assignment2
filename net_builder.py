import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

LAYER_SPACE = dict()
LAYER_SPACE['nb_units'] = (128, 1024, 'int', 0.15)
LAYER_SPACE['dropout_rate'] = (0.0, 0.7, 'float', 0.2)
LAYER_SPACE['activation'] = (0,  ['linear', 'tanh', 'relu', 'sigmoid', 'elu'], 'list', 0.2)

NET_SPACE = dict()
NET_SPACE['nb_layers'] = (1, 3, 'int', 0.15)
NET_SPACE['lr'] = (0.0001, 0.1, 'float', 0.15)
NET_SPACE['weight_decay'] = (0.00001, 0.0004, 'float', 0.2)
NET_SPACE['optimizer'] = (0, ['sgd', 'adam', 'adadelta', 'rmsprop'], 'list', 0.2)

def random_value(space):
   """Sample  random value from the given space."""
   val = None
   if space[2] == 'int':
       val = random.randint(space[0], space[1])
   if space[2] == 'list':
       val = random.sample(space[1], 1)[0]
   if space[2] == 'float':
       val = ((space[1] - space[0]) * random.random()) + space[0]
   return {'val': val, 'id': random.randint(0, 2**10)}


def randomize_network(bounded=False):
   """Create a random network."""
   global NET_SPACE, LAYER_SPACE
   net = dict()
   for k in NET_SPACE.keys():
       net[k] = random_value(NET_SPACE[k])

   if bounded: 
       net['nb_layers']['val'] = min(net['nb_layers']['val'], 1)

   layers = []
   for i in range(net['nb_layers']['val']):
       layer = dict()
       for k in LAYER_SPACE.keys():
           layer[k] = random_value(LAYER_SPACE[k])
       layers.append(layer)
   net['layers'] = layers
   return net

def mutate_net(net):
   """Mutate a network."""
   global NET_SPACE, LAYER_SPACE

   # mutate optimizer
   for k in ['lr', 'weight_decay', 'optimizer']:
       if random.random() < NET_SPACE[k][-1]:
           net[k] = random_value(NET_SPACE[k])
           
   # mutate layers
   for layer in net['layers']:
       for k in LAYER_SPACE.keys():
           if random.random() < LAYER_SPACE[k][-1]:
               layer[k] = random_value(LAYER_SPACE[k])
   # mutate number of layers -- RANDOMLY ADD
   if random.random() < NET_SPACE['nb_layers'][-1]:
       if net['nb_layers']['val'] < NET_SPACE['nb_layers'][1]:
           if random.random()< 0.5:
               layer = dict()
               for k in LAYER_SPACE.keys():
                   layer[k] = random_value(LAYER_SPACE[k])
               net['layers'].append(layer)
               # value & id update
               net['nb_layers']['val'] = len(net['layers'])
               net['nb_layers']['id'] +=1
           else:
               if net['nb_layers']['val'] > 1:
                   net['layers'].pop()
                   net['nb_layers']['val'] = len(net['layers'])
                   net['nb_layers']['id'] -=1
   return net



if __name__ == '__main__':

    
    net = randomize_network()
    # model = CustomModel(net)
