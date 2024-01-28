import statistics
"""
import statistics

def ones_concentration(lst):
    ones_indices = [i for i, x in enumerate(lst) if x == 1]
    if len(ones_indices) == 0:
        return 0.0
    mean_index = statistics.mean(ones_indices)
    stdev_index = statistics.pstdev(ones_indices, mu=mean_index)
    return stdev_index / len(lst)
    
"""


"""
import numpy as np

def get_concentration(lst):
    ones_indices = np.where(np.array(lst) == 1)[0]
    total_ones = len(ones_indices)
    midpoint = len(lst) / 2

    if total_ones < midpoint:
        concentration = (total_ones / midpoint) * 25 + 75
    elif total_ones > midpoint:
        concentration = ((len(lst) - total_ones) / midpoint) * 25 + 75
    else:
        concentration = 50

    if total_ones > 0:
        std_dev = np.std(ones_indices) / len(lst)
        concentration = concentration * (1 - std_dev)

    return concentration
"""
"""
def similarity_score(lst1, lst2):
    set1 = set([i for i, val in enumerate(lst1) if val == 1])
    set2 = set([i for i, val in enumerate(lst2) if val == 1])
    jaccard_distance = 1 - len(set1.intersection(set2)) / len(set1.union(set2))
    return jaccard_distance


# bad=============================good
lst1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# concentration = ones_concentration(lst)
# print(concentration)
lst2 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
print(similarity_score(lst1, lst2))     # 0.9285714285714286

lst3 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
print(similarity_score(lst1, lst3))     # 0.9375
# concentration = ones_concentration(lst2)
# print(concentration)
"""
# import numpy as np
# from scipy.spatial.distance import jaccard
#
# lst1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# lst2 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# lst3 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
#
# array1 = np.array(lst1)
# array2 = np.array(lst2)
# array3 = np.array(lst3)
#
# intersection_1_2 = np.sum(array1 & array2)
# union_1_2 = np.sum(array1 | array2)
# jaccard_distance_1_2 = 1 - intersection_1_2 / union_1_2
#
# intersection_1_3 = np.sum(array1 & array3)
# union_1_3 = np.sum(array1 | array3)
# jaccard_distance_1_3 = 1 - intersection_1_3 / union_1_3
#
# # print(jaccard_distance_1_2)
# # print(jaccard_distance_1_3)
#
# # jaccard_distance1_2 = jaccard(lst1, lst2)
# # print(jaccard_distance1_2)
# # jaccard_distance1_3 = jaccard(lst1, lst3)
# # print(jaccard_distance1_3)
# l_base = [1, 1, 1]
# l = [
#     [0, 0, 0],  # 0
#     [0, 0, 1],  # 0.1
#     [0, 1, 0],  # 0.2
#     [1, 0, 0],  # 0.5
#     [0, 1, 1],  # 0.18
#     [1, 1, 0],  #
#     [1, 0, 1],  #
#     [1, 1, 1]   #
# ]
#
# # 1 - (x & y) / (x | y)
# for item in l:
#     print(item, ": ", jaccard(l_base, item))

import numpy as np

def jaccard_hash_pos(x, y, p=1000003):
    # Compute the Jaccard distance between x and y
    intersection = np.sum(x * y)
    union = np.sum(np.logical_or(x, y))
    jaccard_dist = intersection / union

    # Hash the Jaccard distance value using a large prime number
    hash_val = int(np.floor(jaccard_dist * p))

    # Add a positional component to break ties between similar binary arrays
    pos_weighted_sum = np.sum(x * np.arange(len(x))) + np.sum(y * np.arange(len(y)))
    hash_val = (hash_val + pos_weighted_sum) % p

    return hash_val


# Example usage:
l_base = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
l = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0.1
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0.2
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0.5
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0.18
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   #
]

# 1 - (x & y) / (x | y)
for item in l:
    print(item, ": ", jaccard_hash_pos(l_base, np.array(item)))


"""
<Vector to value>:
1. hashing                                  => Failed - totally random
2. string to int                            => Failed - huge distances
3. Jaccard based on fixed-reference [1's]   => Failed - (collision) same inputs perform 1 output
4. Concentration Point                      => Almost Passed 
"""
