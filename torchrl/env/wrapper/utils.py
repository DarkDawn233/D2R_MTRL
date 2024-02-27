import numpy as np

def update_mean_var_count_from_moments(
        mean, var, count, 
        batch_mean, batch_var, batch_count):
    """
    Imported From OpenAI Baseline
    """
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count