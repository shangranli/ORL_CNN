import numpy as np


def convert_Max_Min(mat):
    h,w = np.shape(mat)

    new_mat = np.empty((h,w))
    for i in range(h):
        for j in range(w):
            new_mat[i,j] = mat[i,j]/255.0


    return new_mat

if __name__ == "main()":
    convert_Max_Min()
