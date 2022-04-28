
import numpy as np
import torch


def score_mutual_information(u, v, to_distance=False):
    """
    Args:
        u (_type_): first vector
        v (_type_): second vector
        to_distance (bool, optional): if True return the mutual distance else return the mutual information 
    """
    
    assert u.shape == v.shape
    m = max(u.max(), v.max()) + 1
    mat = torch.zeros((m, m, m, m))
    for i in range(m):
        for j in range(m):    
            mat[i, j, i, j] = 1
            
    t = torch.cat([u.unsqueeze(2), v.unsqueeze(2)], dim=2)
    
    e = mat[tuple(t.reshape(-1, 2).transpose(1, 0))]    
    counts = e.reshape((t.shape[0], t.shape[1], m, m)).transpose(2, 3).sum(1)
    counts /= u.shape[1]
    p_u = counts.sum(1)
    p_v = counts.sum(2)

    h_u = - (p_u * torch.log2(p_u)).nan_to_num().sum(1)
    h_v = - (p_v * torch.log2(p_v)).nan_to_num().sum(1)
    h_u_v = - (counts * torch.log2(counts)).nan_to_num().sum([1, 2])

    mutual_information = h_u + h_v - h_u_v
    if to_distance:
        m, _ = torch.min(torch.cat((h_u.unsqueeze(1), h_v.unsqueeze(1)), dim=1), 1)
        mutual_information /= m
        return (1 - mutual_information.clip(0, 1)).nan_to_num(np.inf)
    else:
        return mutual_information


def get_model_distance(distance_name):
    if distance_name == "l0":
        return  lambda x, y: (x != y).sum(), "min"
    elif distance_name == "l1":
        return  lambda x, y: (x - y).abs().sum(), "min"
    elif distance_name == "l2":
        return  lambda x, y: (x - y).norm(), "min"
    elif distance_name == "mutual_distance":
        return lambda x, y: score_mutual_information(x, y, to_distance=True), "min"
    elif distance_name == "mutual_information":
        return lambda x, y: score_mutual_information(x, y, to_distance=False), "max"
    else:
        raise ValueError("Unknown Distance:", distance_name)