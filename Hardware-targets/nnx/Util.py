def div_and_ceil(a, b):
    return ((a - 1) // b) + 1

def rem(a, b):
    return ((a - 1) % b) + 1

def maximize_size_w_prio(a, max, prio):
    """Maximize size of argument `a`

    Argument `max` has to be a constant.
    Since division is not allowed in the constraint programming,
    the priority is normalized with the value of `max`.
    """
    return {"value": a, "prio": prio / max}

def maximize_divisibility(a, b):
    return (a - 1) % b

def maximize_divisibility_w_prio(a, b, prio):
    """Maximize divisibility of `a` and `b`

    Second argument `b` has to be a constant.
    """
    return maximize_size_w_prio((a - 1) % b, b - 1, prio)

def maximize_divisibility_or_max_w_prio(a, b, max, prio):
    """Maximize divisibility of `a` and `b` or let `a` be `max`

    Second argument `b` has to be a constant.
    """
    return maximize_size_w_prio((a != max) * ((a - 1) % b) + (a == max) * (b - 1), b - 1, prio)

def maximize_condition(cond, prio):
    """Try to have the condition fulfilled

    If the condition is fulfilled you get the reward equal to `prio`, if not 0.
    """
    return maximize_size_w_prio(cond, 1, prio)

