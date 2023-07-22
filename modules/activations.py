from math import log, exp, tanh
from typing import Callable


Activation = Callable[[float], float]


# activations
def ReLU(x: float) -> float:
    return max(0, x)

def sigmoid(x: float) -> float:
    try:
        return 1 / (1 + exp(-x))
    except OverflowError:
        return 0

def SiLU(x: float) -> float:
    try:
        return x / (1 + exp(-x))
    except OverflowError:
        return 0

def linear(x: float) -> float:
    return x

def softmax(x: list[float]) -> list[float]:
    m = max(x)
    x = [i - m for i in x]
    exps = [exp(i) for i in x]
    inv_total = 1 / sum(exps)
    return [(i) * inv_total for i in exps]

def cross_entropy(observed: list[float], expected: list[float]) -> float:
    return -sum([e * log(o) for o, e in zip(observed, expected)])

# derivatives
def d_tanh(x: float) -> float:
    return 1 - tanh(x) ** 2

def d_ReLU(x: float) -> float:
    return int(x > 0)

def d_sigmoid(x: float) -> float:
    try:
        expx = exp(x)
        return expx / (expx + 1) ** 2
    except OverflowError:
        return 0

def d_SiLU(x: float) -> float:
    try:
        expx = exp(x)
        return expx * (expx + x + 1) / (expx + 1) ** 2
    except OverflowError:
        return 1

def d_linear(x: float) -> float:
    return 1

def d_softmax_cross_entropy(observed: float, expected: float) -> float:
    return observed - expected


activation_derivative: dict[Activation, Activation] = {
    ReLU: d_ReLU,
    SiLU: d_SiLU,
    sigmoid: d_sigmoid,
    tanh: d_tanh,
    linear: d_linear,
}
activation_string: dict[str, Activation] = {
    'ReLU': ReLU,
    'SiLU': SiLU,
    'sigmoid': sigmoid,
    'tanh': tanh,
    'linear': linear,
}
activation_save = {v: k for k, v in activation_string.items()}
