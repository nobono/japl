import numpy as np


def hit_ground_event(t, X, ss, r_targ, *args):
    return X[2]
hit_ground_event.terminal = True


def hit_target_event(t, X, ss, r_targ, *args):
    rm = X[:3]
    # hit_dist = r_targ - rm
    # return  hit_dist[0] + hit_dist[1] + hit_dist[2]
    return  np.linalg.norm(rm)
    
hit_target_event.terminal = True


def check_for_events(t_events):
    for event in t_events:
        if len(event):
            return True
    return False

