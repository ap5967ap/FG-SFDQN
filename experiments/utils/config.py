
import os
import configparser

def reacher_config():
    CONFIG_CONTENT = """
    [GENERAL]
    n_samples=20000
    n_tasks=20
    n_trials=1
    n_batch=512
    buffer_size=200000

    [TASK]
    train_targets=[(0.14, 0.0), (-0.14, 0.0), (0.0, 0.14), (0.0, -0.14)]
    test_targets=[(0.22, 0.0), (-0.22, 0.0), (0.0, 0.22), (0.0, -0.22), (0.1, 0.1), (0.1, -0.1), (-0.1, 0.1), (-0.1, -0.1)]


    [AGENT]
    gamma=0.95
    epsilon=0.60
    T=500
    print_ev=1000
    save_ev=200
    encoding='task'

    [SFQL]
    learning_rate=0.001
    learning_rate_w=0.5
    learning_rate_prior=0.00001
    use_true_reward=False
    hidden_units=128

    [QL]
    learning_rate=0.001

    [FGSF]
    n_averaging=5
    algorithm=alg3
    learning_rate=0.001
    learning_rate_prior=0.00001
    learning_rate_w=0.5
    use_true_reward=False
    hidden_units=128
    """

    if not os.path.exists("configs"):
        os.makedirs("configs")
    with open("configs/config.cfg", "w") as f:
        f.write(CONFIG_CONTENT)


def fourroom_config():
    CONFIG_CONTENT = """
    [GENERAL]
    n_samples=20000
    n_tasks=6
    n_trials=1
    n_batch=512
    buffer_size=200000

    [TASK]
    maze=[
        ['1', ' ', ' ', ' ', ' ', '2', 'X', ' ', ' ', ' ', ' ', ' ', 'G'],
        [' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', '1', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
        ['2', ' ', ' ', ' ', ' ', '3', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
        ['X', 'X', '3', ' ', 'X', 'X', 'X', 'X', 'X', ' ', '1', 'X', 'X'],
        [' ', ' ', ' ', ' ', ' ', ' ', 'X', '2', ' ', ' ', ' ', ' ', '3'],
        [' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', '2', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
        ['_', ' ', ' ', ' ', ' ', ' ', 'X', '3', ' ', ' ', ' ', ' ', '1']]

    [AGENT]
    gamma=0.95
    epsilon=0.60
    T=500
    print_ev=1000
    save_ev=200
    encoding='task'

    [SFQL]
    learning_rate=0.001
    learning_rate_w=0.5
    learning_rate_prior=0.00001
    use_true_reward=False
    hidden_units=128

    [QL]
    learning_rate=0.001

    [FGSF]
    n_averaging=5
    algorithm=alg1
    learning_rate=0.001
    learning_rate_prior=0.00001
    learning_rate_w=0.5
    use_true_reward=False
    hidden_units=128
    """

    if not os.path.exists("configs"):
        os.makedirs("configs")
    with open("configs/config.cfg", "w") as f:
        f.write(CONFIG_CONTENT)

    
def load_config(path="configs/config.cfg"):
    cfg = configparser.ConfigParser()
    cfg.read(path)
    return cfg