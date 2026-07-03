"""Learning 2 Learn utils."""

import numpy as np

import problems


def print_stats(header, total_error, total_time, n):
    """Prints experiment statistics."""
    print(header)
    print("Log Mean Final Error: {:.2f}".format(np.log10(total_error / n)))
    print("Mean epoch time: {:.2f} s".format(total_time / n))


def get_default_net_config(path):
    return {
        "net": "CoordinateWiseDeepLSTM",
        "net_options": {
            "layers": (20, 20),
            "preprocess_name": "LogAndSign",
            "preprocess_options": {"k": 5},
            "scale": 0.01,
        },
        "net_path": path,
    }


def get_config(problem_name, path=None):
    """Returns problem configuration."""
    if problem_name == "simple":
        problem = problems.simple()
        net_config = {"cw": {
            "net": "CoordinateWiseDeepLSTM",
            "net_options": {"layers": (), "initializer": "zeros"},
            "net_path": path,
        }}
        net_assignments = None
    elif problem_name == "simple-multi":
        problem = problems.simple_multi_optimizer()
        net_config = {
            "cw": {
                "net": "CoordinateWiseDeepLSTM",
                "net_options": {"layers": (), "initializer": "zeros"},
                "net_path": path,
            },
            "adam": {
                "net": "Adam",
                "net_options": {"learning_rate": 0.1},
            },
        }
        net_assignments = [("cw", ["x_0"]), ("adam", ["x_1"])]
    elif problem_name == "quadratic":
        problem = problems.quadratic(batch_size=128, num_dims=2)
        net_config = {"cw": {
            "net": "CoordinateWiseDeepLSTM",
            "net_options": {"layers": (20, 20)},
            "net_path": path,
        }}
        net_assignments = None
    elif problem_name == "mnist":
        mode = "train" if path is None else "test"
        problem = problems.mnist(layers=(20,), mode=mode)
        net_config = {"cw": get_default_net_config(path)}
        net_assignments = None
    elif problem_name == "cifar":
        mode = "train" if path is None else "test"
        problem = problems.cifar10("cifar10",
                                   conv_channels=(16, 16, 16),
                                   linear_layers=(32,),
                                   mode=mode)
        net_config = {"cw": get_default_net_config(path)}
        net_assignments = None
    elif problem_name == "square_cos":
        problem = problems.square_cos(batch_size=128, num_dims=2)
        net_config = {"cw": {
            "net": "CoordinateWiseDeepLSTM",
            "net_options": {"layers": (20, 20)},
            "net_path": path,
        }}
        net_assignments = None
    elif problem_name == "protein_dock":
        problem = problems.protein_dock(batch_size=125, num_dims=12)
        net_config = {"cw": {
            "net": "CoordinateWiseDeepLSTM",
            "net_options": {"layers": (20, 20)},
            "net_path": path,
        }}
        net_assignments = None
    else:
        raise ValueError("{} is not a valid problem".format(problem_name))

    return problem, net_config, net_assignments
