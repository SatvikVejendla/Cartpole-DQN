import enum


class Config(enum.Enum):
    alpha=0.001
    gamma = 0.95
    epsilon = 1.0
    decay_rate=0.995
    epsilon_min=0.1
    batch_size=32
    mem_max=2000
    max_ts=100000