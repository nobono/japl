


def hit_ground_event(t, X, ss, r_targ):
    return X[2]
hit_ground_event.terminal = True


def hit_target_event(t, X, ss, r_targ):
    rm = X[:3]
    hit_dist = r_targ - rm
    return  hit_dist[0] + hit_dist[1] + hit_dist[2]
    
hit_target_event.terminal = True

