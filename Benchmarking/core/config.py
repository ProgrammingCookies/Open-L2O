"""
Shared helper for reading required (non-defaulted) config values.

Makes sure all required config values are present and not None, raising a ValueError with a helpful message if they are missing.
Considering how many config values are required for different adapters, this is a useful to mitigate human error
of forgetting to set a parameter and the program defaulting to some other value and running anyways instead of reminding the user to define it.
"""

from __future__ import annotations

from typing import Any, Dict


def require(section: Dict[str, Any], key: str, config_details: str) -> Any:
    """Returns section[key]. Raises ValueError if section[key] is None.

    `config_details` should be a full sentence naming the adapter and the exact config
    path to set, e.g. "l2o-dm requires config['train']['num_epochs']".
    """
    value = section.get(key)
    if value is None:
        raise ValueError(config_details)
    return value
