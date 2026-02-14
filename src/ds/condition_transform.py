"""
Transform user-friendly condition and intervention specs into internal tensor format.

User input format:
- condition: {'digit': '0', 'digit-color': 'red', 'bar-color': 'red', 'bar-width': 'thin'}
- intervention: same format (do(X=x) for causal interventions)

Subsets of keys are allowed. Keys align with full-ncm:
- digit: '0'..'9'
- digit-color: 'red' | 'green'
- bar-color: 'red' | 'green'
- bar-width: 'thin' | 'thick'

Mappings by graph type:
- full-ncm: D=digit, C=digit-color, BC=bar-color, BW=bar-width
- cls-digit: X=digit, B=digit-color, Z=bar-color (no bar-width)
- cls-color: B=digit, X=digit-color (no bar-color, bar-width)
"""

import torch
from typing import Dict, Optional

# Human-readable value -> one-hot index
DIGIT_VALUES = {str(i): i for i in range(10)}
DIGIT_COLOR_VALUES = {"red": 0, "green": 1}
BAR_COLOR_VALUES = {"red": 0, "green": 1}
BAR_WIDTH_VALUES = {"thin": 0, "thick": 1}

# Graph -> (user_key -> internal_var_name)
GRAPH_USER_TO_VAR = {
    "full-ncm": {
        "digit": "D",
        "digit-color": "C",
        "bar-color": "BC",
        "bar-width": "BW",
    },
    "cls-digit": {
        "digit": "X",
        "digit-color": "B",
        "bar-color": "Z",
    },
    "cls-color": {
        "digit": "B",
        "digit-color": "X",
    },
}


def _value_to_onehot(key: str, value: str, graph: str) -> torch.Tensor:
    """Convert a single key-value pair to one-hot tensor."""
    mapping = GRAPH_USER_TO_VAR.get(graph)
    if mapping is None or key not in mapping:
        raise ValueError(
            f"Key '{key}' not supported for graph '{graph}'. "
            f"Valid keys: {list(mapping.keys()) if mapping else 'unknown graph'}"
        )

    if key == "digit":
        size = 10
        idx = DIGIT_VALUES.get(value)
    elif key in ("digit-color", "didgit-color"):
        size = 2
        idx = DIGIT_COLOR_VALUES.get(value)
    elif key == "bar-color":
        size = 2
        idx = BAR_COLOR_VALUES.get(value)
    elif key == "bar-width":
        size = 2
        idx = BAR_WIDTH_VALUES.get(value)
    else:
        raise ValueError(f"Unknown key: {key}")

    if idx is None:
        raise ValueError(
            f"Invalid value '{value}' for key '{key}'. "
            f"Valid: digit=0-9, digit-color={list(DIGIT_COLOR_VALUES)}, "
            f"bar-color={list(BAR_COLOR_VALUES)}, bar-width={list(BAR_WIDTH_VALUES)}"
        )

    onehot = torch.zeros(size)
    onehot[idx] = 1.0
    return onehot


def user_spec_to_internal(
    user_spec: Dict[str, str],
    graph: str,
) -> Dict[str, torch.Tensor]:
    """
    Transform user-friendly condition/intervention dict to internal format.

    Args:
        user_spec: e.g. {'digit': '0', 'digit-color': 'red', 'bar-color': 'red', 'bar-width': 'thin'}
        graph: 'full-ncm' | 'cls-digit' | 'cls-color'

    Returns:
        Dict mapping internal var names to one-hot tensors, e.g. {'D': tensor, 'C': tensor, ...}
    """
    mapping = GRAPH_USER_TO_VAR.get(graph)
    if mapping is None:
        raise ValueError(f"Unknown graph: {graph}. Valid: {list(GRAPH_USER_TO_VAR.keys())}")

    result = {}
    for key, value in user_spec.items():
        key_normalized = key.replace("didgit", "digit")  # typo fix
        if key_normalized not in mapping:
            continue  # skip keys not in this graph
        internal_var = mapping[key_normalized]
        tensor = _value_to_onehot(key_normalized, str(value).strip().lower(), graph)
        result[internal_var] = tensor

    return result


# Default presets when user provides no condition (used for eval)
DEFAULT_CONDITION_DO = {
    "full-ncm": (
        {
            "D": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "C": [1, 0],
            "BC": [1, 0],
            "BW": [1, 0],
        },
        {"D": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]},
    ),
    "cls-digit": (
        {
            "X": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "B": [1, 0],
            "Z": [1, 0],
        },
        {"X": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]},
    ),
    "cls-color": (
        {
            "B": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "X": [0, 1],
        },
        {"X": [1, 0]},
    ),
}


def parse_condition_intervention_args(
    condition_str: Optional[str],
    intervention_str: Optional[str],
    graph: str,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Parse condition and intervention from CLI-style strings.

    Format: "key1=val1,key2=val2" e.g. "digit=0,digit-color=red,bar-width=thin"

    When condition is not provided, returns graph-specific defaults.

    Returns:
        (condition dict, intervention dict)
    """
    def parse_kv(s: Optional[str]) -> Dict[str, str]:
        if not s or not s.strip():
            return {}
        out = {}
        for part in s.split(","):
            part = part.strip()
            if "=" in part:
                k, v = part.split("=", 1)
                out[k.strip()] = v.strip()
        return out

    cond_user = parse_kv(condition_str)
    do_user = parse_kv(intervention_str)

    condition = user_spec_to_internal(cond_user, graph) if cond_user else {}
    do = user_spec_to_internal(do_user, graph) if do_user else {}

    if not condition:
        preset = DEFAULT_CONDITION_DO.get(graph)
        if preset is None:
            raise ValueError(f"Unknown graph: {graph}. Valid: {list(DEFAULT_CONDITION_DO.keys())}")
        cond_default, do_default = preset
        condition = {k: torch.tensor(v) for k, v in cond_default.items()}
        do = do or {k: torch.tensor(v) for k, v in do_default.items()}

    return condition, do
