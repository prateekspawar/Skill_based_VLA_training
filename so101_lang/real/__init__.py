"""Real SO-101 host, run on the Raspberry Pi next to the hardware (copied there as ``~/so101_real``).

``arm_host.py`` owns the servo bus and holds or moves the arm on request; ``arm_cmd.py`` is its client;
``arm_config.json`` sets the per-joint gains and safety limits, applied every time the host starts. These files
are standalone (Feetech bus code from LeRobot only) because the Pi does not install this package.
"""
