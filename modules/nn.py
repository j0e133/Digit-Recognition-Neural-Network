from typing import overload
from json import dumps, loads
from random import uniform
from modules.activations import *


class DataPoint:
    __slots__ = ('inputs', 'expected')

    def __init__(self, inputs: list[float], expected: list[float]) -> None:
        self.inputs = inputs
        self.expected = expected


class Layer:
    __slots__ = ('nodes_in', 'nodes_out', 'weights', 'biases', 'activation_function', 'activation_function_derivative', 'weighted_inputs', 'activations', 'weight_cost_gradient', 'bias_cost_gradient', 'inputs')

    @overload
    def __init__(self, n_in: int, n_out: int, activation_function: Activation): ...
    @overload
    def __init__(self, data: tuple[int, int, list[list[float]], list[float], str]): ...
    def __init__(self, *args: int | Activation | tuple[int, int, list[list[float]], list[float], str], **kwargs):
        if isinstance(args[0], int) and isinstance(args[1], int) and isinstance(args[2], Callable):
            n_in = args[0]
            n_out = args[1]
            activation_function = args[2]

            self.nodes_in = n_in
            self.nodes_out = n_out

            self.weights: list[list[float]] = []
            self.biases: list[float] = [0 for _ in range(n_out)]

            self.activation_function = activation_function
            self.activation_function_derivative = activation_derivative[activation_function]

            self.initialize_random_weights()

        else:
            data: tuple[int, int, list[list[float]], list[float], str] = args[0] # type: ignore

            self.nodes_in = data[0]
            self.nodes_out = data[1]

            self.weights = data[2]
            self.biases = data[3]
            self.activation_function = activation_string[data[4]]
            self.activation_function_derivative = activation_derivative[self.activation_function]

        self.weighted_inputs: list[float] = [0 for _ in range(self.nodes_out)]
        self.activations: list[float] = [0 for _ in range(self.nodes_out)]

        self.weight_cost_gradient: list[list[float]] = [[0 for _ in range(self.nodes_in)] for _ in range(self.nodes_out)]
        self.bias_cost_gradient: list[float] = [0 for _ in range(self.nodes_out)]

    def calculate_outputs(self, inputs: list[float]) -> list[float]:
        self.inputs = inputs

        for node_out in range(self.nodes_out):
            weighted_input = sum([i * w for i, w in zip(inputs, self.weights[node_out])], self.biases[node_out])
            self.weighted_inputs[node_out] = weighted_input
            self.activations[node_out] = self.activation_function(weighted_input)

        return self.activations

    def update_gradients(self, node_values: list[float]) -> None:
        for node_out in range(self.nodes_out):
            self.bias_cost_gradient[node_out] += node_values[node_out]

            for node_in in range(self.nodes_in):
                self.weight_cost_gradient[node_out][node_in] += self.inputs[node_in] * node_values[node_out]

    def apply_gradients(self, learn_rate: float) -> None:
        for node_out in range(self.nodes_out):
            self.biases[node_out] -= self.bias_cost_gradient[node_out] * learn_rate
            self.bias_cost_gradient[node_out] = 0

            for node_in in range(self.nodes_in):
                self.weights[node_out][node_in] -= self.weight_cost_gradient[node_out][node_in] * learn_rate
                self.weight_cost_gradient[node_out][node_in] = 0

    def calculate_output_layer_node_values(self, expected: list[float]) -> list[float]:
        return [d_softmax_cross_entropy(o, e) for o, e in zip(self.activations, expected)]

    def calculate_hidden_layer_node_values(self, previous_layer: 'Layer', previous_node_values: list[float]) -> list[float]:
        return [sum([previous_layer_weights[next_node_index] * previous_node_value for previous_layer_weights, previous_node_value in zip(previous_layer.weights, previous_node_values)]) * self.activation_function_derivative(self.weighted_inputs[next_node_index]) for next_node_index in range(self.nodes_out)]

    def process(self, inputs: list[float]) -> list[float]:
        return [self.activation_function(sum([i * w for i, w in zip(inputs, self.weights[node_out])], self.biases[node_out]))for node_out in range(self.nodes_out)]
    
    def process_output_layer(self, inputs: list[float]) -> list[float]:
        return softmax([self.activation_function(sum([i * w for i, w in zip(inputs, self.weights[node_out])], self.biases[node_out]))for node_out in range(self.nodes_out)])

    def initialize_random_weights(self) -> None:
        scale = 1 / self.nodes_in
        for node_out in range(self.nodes_out):
            self.weights.append([])
            for node_in in range(self.nodes_in):
                self.weights[node_out].append(uniform(-1, 1) * scale)

    def save(self) -> tuple[int, int, list[list[float]], list[float], str]:
        return (self.nodes_in, self.nodes_out, self.weights, self.biases, activation_save[self.activation_function])


class OutputLayer(Layer):
    def calculate_outputs(self, inputs: list[float]) -> list[float]:
        self.inputs = inputs

        for node_out in range(self.nodes_out):
            weighted_input = sum([i * w for i, w in zip(inputs, self.weights[node_out])], self.biases[node_out])
            self.weighted_inputs[node_out] = weighted_input
            self.activations[node_out] = self.activation_function(weighted_input)

        return softmax(self.activations)

    def process(self, inputs: list[float]) -> list[float]:
        return softmax([self.activation_function(sum([i * w for i, w in zip(inputs, self.weights[node_out])], self.biases[node_out])) for node_out in range(self.nodes_out)])


class NeuralNetwork:
    __slots__ = ('layers')

    @overload
    def __init__(self, layer_sizes: list[int], layer_activation_functions: list[Activation]): ...
    @overload
    def __init__(self, filename: str): ...
    def __init__(self, *args, **kwargs):
        if isinstance(args[0], str):
            filename = args[0]

            with open(filename, 'r') as f:
                json: list[tuple[int, int, list[list[float]], list[float], str]] = loads(f.read())

            self.layers: list[Layer] = []

            for layer_data in json[:-1]:
                self.layers.append(Layer(layer_data))
            self.layers.append(OutputLayer(json[-1]))

        else:
            layer_sizes: list[int] = args[0]
            layer_activation_functions: list[Activation] = args[1]

            self.layers: list[Layer] = []

            for i in range(len(layer_sizes) - 2):
                self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1], layer_activation_functions[i]))
            self.layers.append(OutputLayer(layer_sizes[len(layer_sizes) - 2], layer_sizes[len(layer_sizes) - 1], layer_activation_functions[len(layer_sizes) - 2]))

    def calculate_outputs(self, inputs: list[float]) -> None:
        for layer in self.layers:
            inputs = layer.calculate_outputs(inputs)

    def apply_all_gradients(self, learn_rate: float) -> None:
        for layer in self.layers:
            layer.apply_gradients(learn_rate)

    def update_all_gradients(self, data_point: DataPoint) -> None:
        self.calculate_outputs(data_point.inputs)

        output_layer = self.layers[-1]
        node_values = output_layer.calculate_output_layer_node_values(data_point.expected)
        output_layer.update_gradients(node_values)

        for hidden_layer_index in range(len(self.layers) - 2, -1, -1):
            layer = self.layers[hidden_layer_index]
            next_layer = self.layers[hidden_layer_index + 1]
            node_values = layer.calculate_hidden_layer_node_values(next_layer, node_values)
            layer.update_gradients(node_values)

    def train(self, training_batch: list[DataPoint], learn_rate: float) -> None:
        for data_point in training_batch:
            self.update_all_gradients(data_point)

        self.apply_all_gradients(learn_rate / len(training_batch))

    def process(self, inputs: list[float]) -> list[float]:
        for layer in self.layers:
            inputs = layer.process(inputs)

        return inputs

    def error(self, data_set: list[DataPoint]) -> float:
        return sum([cross_entropy(self.process(data_point.inputs), data_point.expected) for data_point in data_set]) / len(data_set)

    def save(self, filename: str) -> None:
        data = [layer.save() for layer in self.layers]
        json = dumps(data)

        with open(filename, 'w') as f:
            f.write(json)
