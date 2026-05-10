import datetime
import json
import os
import re
import subprocess
import threading

_L2O_PATH_RE = re.compile(
    r'(DM_enhanced|DM|RNNProp)_lasso-([0-9.]+)_m([0-9]+)_n([0-9]+)(?:_noise([0-9.]+))?'
)
_L2O_NAME_MAP = {'DM_enhanced': 'l2o_dm_enhanced', 'DM': 'l2o_dm', 'RNNProp': 'l2o_rnnprop'}


def _parse_l2o_save_path(save_path):
    """Parse (model_name, lam, m, n, noise_std) from an L2O save_path directory name."""
    match = _L2O_PATH_RE.search(os.path.basename(save_path))
    if match is None:
        return 'l2o_unknown', None, None, None, 0.0
    tag, lam, dim_m, dim_n, noise = match.groups()
    return (_L2O_NAME_MAP[tag], float(lam), int(dim_m), int(dim_n),
            float(noise) if noise else 0.0)


def save_l2o_profile(profile_dir, save_path, sampler, elapsed_s, num_steps):
    """Append one profile entry to training_profiles.jsonl, parsing metadata from save_path."""
    model_name, lam, m, n, noise_std = _parse_l2o_save_path(save_path)
    mean_gpu_mb, median_gpu_mb, n_samples = sampler.stats()
    profile = {
        'timestamp': datetime.datetime.now().isoformat(),
        'model_name': model_name,
        'num_layers': num_steps,
        'lasso_lam': lam,
        'm': m,
        'n': n,
        'noise_std': noise_std,
        'num_train_images': None,
        'train_batch_size': None,
        'training_time_s': elapsed_s,
        'gpu_mem_mean_mb': mean_gpu_mb,
        'gpu_mem_median_mb': median_gpu_mb,
        'gpu_mem_n_samples': n_samples,
    }
    os.makedirs(profile_dir, exist_ok=True)
    profile_path = os.path.join(profile_dir, 'training_profiles.jsonl')
    with open(profile_path, 'a') as f:
        f.write(json.dumps(profile) + '\n')
    print("Profile saved to {}".format(profile_path))


class GpuMemSampler:
    """Polls nvidia-smi in a background thread and collects GPU memory-used samples (MB).

    Uses the GPU-wide memory query, which works on Windows WDDM drivers.
    Per-process memory is not available via nvidia-smi on WDDM (consumer GPUs).
    """

    def __init__(self, interval_s=10.0):
        self._interval = interval_s
        self._samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()

    def _loop(self):
        while not self._stop.wait(self._interval):
            mb = self._read()
            if mb is not None:
                self._samples.append(mb)

    def _read(self):
        try:
            out = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.used',
                 '--format=csv,noheader,nounits'],
                stderr=subprocess.DEVNULL, timeout=5)
            return float(out.decode().strip().split('\n')[0])
        except Exception:
            return None

    def stats(self):
        s = sorted(self._samples)
        if not s:
            return None, None, 0
        mean = round(sum(s) / len(s), 1)
        mid = len(s) // 2
        median = s[mid] if len(s) % 2 else round((s[mid - 1] + s[mid]) / 2, 1)
        return mean, median, len(s)
