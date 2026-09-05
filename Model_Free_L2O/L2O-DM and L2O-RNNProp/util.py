# Copyright 2016 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
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


def get_config(problem_name, path=None, mode=None, num_hidden_layer=None, net_name=None,
              lasso_data_dir=None, lasso_split="train_data.npy", lasso_batch_size=128,
              lasso_lam=0.005, lasso_x0_mode="aligned"):
    """Returns (problem_build_fn, net_config, net_assignments)."""
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
                "net_options": {"learning_rate": 0.01},
            },
        }
        net_assignments = [("cw", ["x_0"]), ("adam", ["x_1"])]

    elif problem_name == "quadratic":
        problem = problems.quadratic(batch_size=128, num_dims=10)
        net_config = {"cw": {
            "net": "CoordinateWiseDeepLSTM",
            "net_options": {"layers": (20, 20)},
            "net_path": path,
        }}
        net_assignments = None

    elif problem_name == "mnist":
        if mode is None:
            mode = "train" if path is None else "test"
        problem = problems.mnist(layers=(20,), activation="sigmoid", mode=mode)
        net_config = {"cw": get_default_net_config(path)}
        net_assignments = None

    elif problem_name == "mnist_relu":
        if mode is None:
            mode = "train" if path is None else "test"
        problem = problems.mnist(layers=(20,), activation="relu", mode=mode)
        net_config = {"cw": get_default_net_config(path)}
        net_assignments = None

    elif problem_name == "mnist_deeper":
        if mode is None:
            mode = "train" if path is None else "test"
        problem = problems.mnist(layers=(20, 20), activation="sigmoid", mode=mode)
        net_config = {"cw": get_default_net_config(path)}
        net_assignments = None

    elif problem_name == "mnist_conv":
        if mode is None:
            mode = "train" if path is None else "test"
        problem = problems.mnist_conv(mode=mode, batch_norm=True)
        net_config = {"cw": get_default_net_config(path)}
        net_assignments = None

    elif problem_name == "cifar_conv":
        if mode is None:
            mode = "train" if path is None else "test"
        problem = problems.cifar10(mode=mode)
        net_config = {"cw": get_default_net_config(path)}
        net_assignments = None

    elif problem_name == "lenet":
        if mode is None:
            mode = "train" if path is None else "test"
        problem = problems.LeNet(conv_channels=(6, 16), linear_layers=(120, 84), mode=mode)
        net_config = {"cw": get_default_net_config(path)}
        net_assignments = None

    elif problem_name == "nas":
        if mode is None:
            mode = "train" if path is None else "test"
        problem = problems.NAS(mode=mode)
        net_config = {"cw": get_default_net_config(path)}
        net_assignments = None

    elif problem_name == "vgg16":
        if mode is None:
            mode = "train" if path is None else "test"
        problem = problems.vgg16_cifar10(mode=mode)
        net_config = {"cw": get_default_net_config(path)}
        net_assignments = None

    elif problem_name == "confocal_microscopy_3d":
        problem = problems.confocal_microscopy_3d(batch_size=32, num_points=5)
        net_config = {"cw": {
            "net": "CoordinateWiseDeepLSTM",
            "net_options": {"layers": (20, 20)},
            "net_path": path,
        }}
        net_assignments = None

    elif problem_name == "square_cos":
        problem = problems.square_cos(batch_size=128, num_dims=2)
        net_config = {"cw": {
            "net": "CoordinateWiseDeepLSTM",
            "net_options": {"layers": (20, 20)},
            "net_path": path,
        }}
        net_assignments = None

    elif problem_name == "rastrigin":
        problem = problems.rastrigin(batch_size=128, num_dims=2)
        net_config = {"cw": {
            "net": "CoordinateWiseDeepLSTM",
            "net_options": {"layers": (20, 20)},
            "net_path": path,
        }}
        net_assignments = None

    elif problem_name == "lasso":
        problem = problems.lasso(batch_size=128, num_dims=2)
        net_config = {"cw": {
            "net": "CoordinateWiseDeepLSTM",
            "net_options": {"layers": (20, 20)},
            "net_path": path,
        }}
        net_assignments = None

    elif problem_name == "lasso_dataset":
        if lasso_data_dir is None:
            raise ValueError("problem 'lasso_dataset' requires --lasso_data_dir")
        problem = problems.lasso_from_dataset(
            lasso_data_dir, split=lasso_split, batch_size=lasso_batch_size, l=lasso_lam,
            deterministic=(mode == "test"), x0_mode=lasso_x0_mode)
        net_config = {"cw": {
            "net": "CoordinateWiseDeepLSTM",
            "net_options": {"layers": (20, 20)},
            "net_path": path,
        }}
        net_assignments = None

    else:
        raise ValueError("{} is not a valid problem".format(problem_name))

    if net_name == "RNNprop":
        default_config = {
            "net": "RNNprop",
            "net_options": {
                "layers": (20, 20),
                "preprocess_name": "fc",
                "preprocess_options": {"dim": 20},
                "scale": 0.01,
                "tanh_output": True,
            },
            "net_path": path,
        }
        net_config = {"rp": default_config}

    return problem, net_config, net_assignments
