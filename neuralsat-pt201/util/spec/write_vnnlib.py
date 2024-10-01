import torch

def write_vnnlib(spec_path: str,  
                 data_lb: float, data_ub: float, 
                 prediction: torch.Tensor,
                 negate_spec=False) -> str:
    # input bounds
    x_lb = data_lb.flatten()
    x_ub = data_ub.flatten()
    
    # outputs
    n_class = prediction.numel()
    y = prediction.argmax(-1).item()
    
    with open(spec_path, "w") as f:
        f.write(f"; Specification for class {int(y)}\n")

        f.write(f"\n; Definition of input variables\n")
        for i in range(len(x_ub)):
            f.write(f"(declare-const X_{i} Real)\n")

        f.write(f"\n; Definition of output variables\n")
        for i in range(n_class):
            f.write(f"(declare-const Y_{i} Real)\n")

        f.write(f"\n; Definition of input constraints\n")
        for i in range(len(x_ub)):
            f.write(f"(assert (<= X_{i} {x_ub[i]:.8f}))\n")
            f.write(f"(assert (>= X_{i} {x_lb[i]:.8f}))\n\n")

        f.write(f"\n; Definition of output constraints\n")
        if not negate_spec:
            for i in range(n_class):
                if i == y:
                    continue
                f.write(f"(assert (<= Y_{i} Y_{y}))\n")
        else:
            f.write(f"(assert (or\n")
            for i in range(n_class):
                if i == y:
                    continue
                f.write(f"\t(and (>= Y_{i} Y_{y}))\n")
            f.write(f"))\n")
    return spec_path
