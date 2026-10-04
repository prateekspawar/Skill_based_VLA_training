from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors


def make_smolvla_depth_pre_post_processors(config, dataset_stats=None):
    return make_smolvla_pre_post_processors(config, dataset_stats)
