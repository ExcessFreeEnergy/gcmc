"""
PufferLib-compatible reinforcement learning environment for cDFT fluid manipulation.
Accelerated with native C core (libcdft_env.so).
"""

import os
import ctypes
from ctypes import c_void_p, c_float, c_int, c_ubyte, Structure, POINTER
import numpy as np

try:
    import gymnasium
    from gymnasium import spaces
except ImportError:
    # Minimal fallback space implementation if gymnasium is not in environment
    class spaces:
        class Box:
            def __init__(self, low, high, shape, dtype=np.float32):
                self.low = np.full(shape, low, dtype=dtype) if np.isscalar(low) else np.array(low, dtype=dtype)
                self.high = np.full(shape, high, dtype=dtype) if np.isscalar(high) else np.array(high, dtype=dtype)
                self.shape = tuple(shape)
                self.dtype = dtype

            def sample(self):
                return np.random.uniform(self.low, self.high, size=self.shape).astype(self.dtype)

# Path to shared C library
_DIR = os.path.dirname(os.path.abspath(__file__))
_SO_PATH = os.path.join(_DIR, "libcdft_env.so")


def _load_c_lib():
    if not os.path.exists(_SO_PATH):
        import subprocess
        cmd = f"gcc -O3 -shared -fPIC -lm {_DIR}/cdft_env.c -o {_SO_PATH}"
        subprocess.run(cmd, shell=True, check=True)
    return ctypes.CDLL(_SO_PATH)


_clib = _load_c_lib()

CDFT_GRID_SIZE = 50
CDFT_NUM_ACTIONS = 3
CDFT_OBS_SIZE = CDFT_GRID_SIZE * 2 + 6


class CLog(Structure):
    _fields_ = [
        ("score", c_float),
        ("n", c_float),
    ]


class CCdftEnv(Structure):
    _fields_ = [
        ("log", CLog),
        ("observations", POINTER(c_float)),
        ("actions", POINTER(c_float)),
        ("rewards", POINTER(c_float)),
        ("terminals", POINTER(c_ubyte)),
        ("tick", c_int),
        ("max_ticks", c_int),
        ("T", c_float),
        ("mu", c_float),
        ("L_slit", c_float),
        ("target_theta", c_float),
        ("rho", c_float * CDFT_GRID_SIZE),
        ("charge_n", c_float * CDFT_GRID_SIZE),
        ("phi0", c_float),
        ("mode_m", c_float),
        ("v_bias", c_float),
    ]


_clib.c_reset.argtypes = [POINTER(CCdftEnv)]
_clib.c_step.argtypes = [POINTER(CCdftEnv)]
_clib.c_render.argtypes = [POINTER(CCdftEnv)]
_clib.c_close.argtypes = [POINTER(CCdftEnv)]


class CdftFluidEnv:
    """
    Continuous control environment for cDFT dielectrocapillary fluid manipulation.
    """
    def __init__(self, max_ticks=100, seed=None):
        self.single_observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(CDFT_OBS_SIZE,), dtype=np.float32
        )
        self.single_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(CDFT_NUM_ACTIONS,), dtype=np.float32
        )
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space
        self.num_agents = 1

        self._obs_buf = np.zeros(CDFT_OBS_SIZE, dtype=np.float32)
        self._act_buf = np.zeros(CDFT_NUM_ACTIONS, dtype=np.float32)
        self._rew_buf = np.zeros(1, dtype=np.float32)
        self._term_buf = np.zeros(1, dtype=np.uint8)

        self._c_env = CCdftEnv()
        self._c_env.observations = self._obs_buf.ctypes.data_as(POINTER(c_float))
        self._c_env.actions = self._act_buf.ctypes.data_as(POINTER(c_float))
        self._c_env.rewards = self._rew_buf.ctypes.data_as(POINTER(c_float))
        self._c_env.terminals = self._term_buf.ctypes.data_as(POINTER(c_ubyte))
        self._c_env.max_ticks = max_ticks

        if seed is not None:
            np.random.seed(seed)

    def reset(self, seed=None, options=None):
        _clib.c_reset(ctypes.byref(self._c_env))
        info = {
            "T": self._c_env.T,
            "mu": self._c_env.mu,
            "target_theta": self._c_env.target_theta,
        }
        return self._obs_buf.copy(), info

    def step(self, action):
        self._act_buf[:] = action
        _clib.c_step(ctypes.byref(self._c_env))

        obs = self._obs_buf.copy()
        reward = float(self._rew_buf[0])
        terminated = bool(self._term_buf[0])
        truncated = False
        info = {
            "score": float(self._c_env.log.score),
            "phi0": float(self._c_env.phi0),
            "mode_m": float(self._c_env.mode_m),
            "target_theta": float(self._c_env.target_theta),
        }
        return obs, reward, terminated, truncated, info

    @property
    def target_filling(self):
        return float(self._c_env.target_theta)

    @target_filling.setter
    def target_filling(self, value):
        self._c_env.target_theta = float(value)

    @property
    def density_profile(self):
        return np.array(self._c_env.rho, dtype=np.float32)

    @property
    def charge_profile(self):
        return np.array(self._c_env.charge_n, dtype=np.float32)

    @property
    def current_filling(self):
        return float(np.mean(self.density_profile))

    @property
    def phi_0(self):
        return float(self._c_env.phi0)

    @phi_0.setter
    def phi_0(self, val):
        self._c_env.phi0 = float(val)

    @property
    def mode_m(self):
        return float(self._c_env.mode_m)

    @mode_m.setter
    def mode_m(self, val):
        self._c_env.mode_m = float(val)

    @property
    def v_bias(self):
        return float(self._c_env.v_bias)

    @v_bias.setter
    def v_bias(self, val):
        self._c_env.v_bias = float(val)

    def render(self):
        _clib.c_render(ctypes.byref(self._c_env))

    def close(self):
        _clib.c_close(ctypes.byref(self._c_env))


class BatchedCdftVecEnv:
    """
    High-throughput vectorized environment running N independent cDFT instances
    in contiguous memory buffers.
    """
    def __init__(self, num_envs=64, max_ticks=100):
        self.num_envs = num_envs
        self.single_observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(CDFT_OBS_SIZE,), dtype=np.float32
        )
        self.single_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(CDFT_NUM_ACTIONS,), dtype=np.float32
        )

        self.observations = np.zeros((num_envs, CDFT_OBS_SIZE), dtype=np.float32)
        self.actions = np.zeros((num_envs, CDFT_NUM_ACTIONS), dtype=np.float32)
        self.rewards = np.zeros(num_envs, dtype=np.float32)
        self.terminals = np.zeros(num_envs, dtype=np.uint8)

        self._c_envs = (CCdftEnv * num_envs)()
        for i in range(num_envs):
            c_env = self._c_envs[i]
            c_env.observations = self.observations[i].ctypes.data_as(POINTER(c_float))
            c_env.actions = self.actions[i].ctypes.data_as(POINTER(c_float))
            c_env.rewards = self.rewards[i:i+1].ctypes.data_as(POINTER(c_float))
            c_env.terminals = self.terminals[i:i+1].ctypes.data_as(POINTER(c_ubyte))
            c_env.max_ticks = max_ticks
            _clib.c_reset(ctypes.byref(c_env))

    def reset(self):
        for i in range(self.num_envs):
            _clib.c_reset(ctypes.byref(self._c_envs[i]))
        return self.observations.copy(), {}

    def step(self, actions):
        self.actions[:] = actions
        for i in range(self.num_envs):
            _clib.c_step(ctypes.byref(self._c_envs[i]))
            if self.terminals[i]:
                _clib.c_reset(ctypes.byref(self._c_envs[i]))

        return self.observations.copy(), self.rewards.copy(), self.terminals.copy(), {}
