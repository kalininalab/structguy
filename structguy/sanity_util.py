import numpy as np

def sanity_check_value_list(values, label_vector = None, datastructure_name = 'placeholder'):
    insane_pos = []
    for pos, value in enumerate(values):
        try:
            isfinite = np.isfinite(value)
        except:
            isfinite = False
        if not isfinite:
            insane_pos.append(pos)

    if len(insane_pos) == 0:
        return None
    if label_vector is not None:
        for pos in insane_pos:
            print(f'Detected insane value in {datastructure_name}. At position {pos}, value: {values[pos]}, value type: {type(values[pos])}, label: {label_vector[pos]}')
    return insane_pos