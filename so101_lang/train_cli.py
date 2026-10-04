"""``lerobot-train`` with this project's policy plugins registered (e.g. ``--policy.type=smolvla_depth``).

    ~/lerobot-env/bin/python -m so101_lang.train_cli <lerobot-train arguments>
"""

import so101_lang.policies.smolvla_depth  # noqa: F401  (registers the smolvla_depth policy)
from lerobot.scripts.lerobot_train import main

if __name__ == "__main__":
    main()
