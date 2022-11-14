from datetime import datetime
import os


def prepare_checkpoint_path(cp_dir, experiment_name):
    """
    Create directories if not exist.
    """
    os.makedirs(cp_dir, exist_ok=True)
    train_id = experiment_name + "-" + datetime.now().strftime(
        "%Y_%m_%d-%H_%M_%S")
    return os.path.join(cp_dir, train_id), train_id
