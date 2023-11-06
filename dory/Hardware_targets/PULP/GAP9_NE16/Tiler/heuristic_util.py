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

def minimize_size_w_prio(a, max, prio):
    """Minimize the value of `a`

    Since we are maximizing our objective function, minimizing a value is the
    same problem as maximizing the negation of the value. To keep the value
    in the positive range, it is subtracted from the maximal value.
    """
    return maximize_size_w_prio(max-a, max, prio)

def heuristic_sum(heuristics, modifier=1000000):
    return sum([int(modifier * h["prio"]) * h["value"] for h in heuristics])
