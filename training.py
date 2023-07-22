from modules.nn import *
from modules.mnist import MNIST


dataset = MNIST()
print('mnist loaded')
network = NeuralNetwork('saves/digit_recognition.json')
print('network loaded')

print(f'Starting error: {network.error(dataset.get_random_images(750)):.6f}')

batch_size = 500
batches = 10_000
starting_generation = 1125 + 1

for i in range(batches):
    print(f'Training batch {i + starting_generation}')

    network.train(dataset.get_random_images(batch_size), 0.0001)
    network.save('saves/digit_recognition.json')

    print(f'Finished training batch {i + starting_generation}')

    if (i + starting_generation) % 50 == 0:
        network.save(f'saves/digit_recognition{i + starting_generation}.json')
        print(f'Current error: {network.error(dataset.get_random_images(1000)):.6f}')

