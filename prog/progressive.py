import numpy as np


def progressive_schedule(args, r_max=256, l_max=4):
    num_stages = args.num_stages
    r_scale = args.r_scale
    l_scale = args.l_scale


    train_spochs = args.epochs
    epoch_list = [int(i) for i in np.linspace(0, train_spochs, num_stages + 1) // 1][:-1]
    r_list = [make_divisible(i, 32) for i in np.linspace(r_scale, 1., num_stages) * r_max]
    l_list = [make_divisible(i, 1) for i in np.linspace(l_scale, 1., num_stages) * l_max]

    return epoch_list, r_list, l_list


def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v
