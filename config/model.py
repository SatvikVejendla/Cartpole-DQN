import enum


class Structure(enum.Enum):
    i_dim = 4
    h1 = 12
    a1="relu"
    h2 = 8
    a2="relu"
    o = 2
    loss="mse"