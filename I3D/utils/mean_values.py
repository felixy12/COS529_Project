# Source: https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/mean.py

def get_mean(norm_value=255, dataset='ufc101'):

    # Below values are in RGB order
    return [101.00131/norm_value, 97.3644226/norm_value, 89.42114168/norm_value]

def get_std(norm_value=255):
    # Kinetics (10 videos for each class)
    return [38.7568578/norm_value, 37.88248729/norm_value, 40.02898126/norm_value]
