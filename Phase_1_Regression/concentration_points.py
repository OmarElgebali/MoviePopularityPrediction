def w_avg(arr):
    weight = 0  # weight
    s = 0       # position*weight
    for element in arr:
        s += (element[0] * element[1])  # s += index * weight
        weight += element[1]
    return s/weight     # weighted average


def split_arr(arr, n_splits):
    # looping till length l
    for i in range(0, len(arr), n_splits):
        yield arr[i:i + n_splits]


def find_concentration(arr, n=5):  # n is the number of concentration points to find
    # seperate array into batches
    batches = list(split_arr(arr, int(len(arr) / n)))
    concentrations = []
    for i in range(len(batches)):
        point = 0
        num_ones = 0
        for j in range(len(batches[i])):
            if batches[i][j] == 1:
                point += j + (i * int(len(arr) / n))  # adding correction for batches
                num_ones += 1
        if num_ones > 0:
            point = point / num_ones
            concentrations.append((point, num_ones))
    return concentrations

# arr1 = [1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0,
#         1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0,
#         1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1,
#         1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0]

# arr1 = [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0,
#         1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

arr2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0, 0, 0]

arr3 = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 0, 0, 0, 1]

# print(w_avg(find_concentration(arr1)))

print(find_concentration(arr2))
print(w_avg(find_concentration(arr2)))
print('='*100)
print(find_concentration(arr3))
print(w_avg(find_concentration(arr3)))
