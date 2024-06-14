

def has_relu_var(model):
    names = [v.VarName for v in model.getVars()]
    if not len(names):
        return False
    relus = [v for v in names if v.startswith('aReLU')]
    return len(relus) > 0
