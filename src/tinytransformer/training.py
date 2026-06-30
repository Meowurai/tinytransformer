

def zero_grad(model):
    for param in model.parameters():
        param.grad = 0.0

def step(model, learning_rate: float):
    for parameter in model.parameters():
        parameter.data -= learning_rate * parameter.grad