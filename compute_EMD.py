import numpy as np
import csv
import generate_lp_file
import sys, getopt


def read_results(filename_f: str) -> np.ndarray:
    '''
    Parse the .csv export from LPSolve IDE to get the solution F
    :param filename_f:
    :return:
    '''
    f = open(filename_f, "r")
    reader = csv.reader((x.replace('\0', '') for x in f), delimiter=";")
    i_list, j_list = [], []
    for k, row in enumerate(reader):
        if k > 1:
            if len(row) > 0:
                index = row[0].split(".")
                (i, j) = (int(index[0][1:]), int(index[1]))
                i_list.append(i)
                j_list.append(j)
    m = max(i_list)+1
    n = max(j_list)+1
    f.close()

    f = open(filename_f, "r")
    reader = csv.reader((x.replace('\0', '') for x in f), delimiter=";")
    f_mat = np.zeros((m, n))
    for k, row in enumerate(reader):
        if k > 1:
            if len(row) > 0:
                index = row[0].split(".")
                (i, j) = (int(index[0][1:]), int(index[1]))
                if i < m and j < n:
                    f_mat[i, j] = float(row[1].replace(',', '.'))
    f.close()
    return f_mat


def read_d(filename: str) -> np.ndarray:
    '''
    reads the .npy file with the distances between edges
    :param filename:
    :return:
    '''
    d = np.load(filename)
    return d


def compute_emd(d_mat: np.ndarray,
                f_mat: np.ndarray):
    '''
    computes the final EMD
    :param d_mat: distance across edges from P to Q
    :param f_mat: optimal flow (solution of the linear programming problem)
    :return:
    '''
    return np.trace(np.dot(d_mat.T, f_mat))


if __name__ == '__main__':
    FILENAME_F = "model.lp.csv"
    FILENAME_D = generate_lp_file.FILENAME_D

    # We parse the arguments for script execution in the console
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "f:c:", ["filename_csv=", "filename_d="])
    except getopt.GetoptError:
        print(argv)
        print('argument parsing error')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-f', '--filename_csv'):
            FILENAME_MODEL = arg
        elif opt in ('-c', '--filename_d'):
            FILENAME_D = arg

    d_mat = read_d(FILENAME_D)
    f_mat = read_results(FILENAME_F)

    print("EMD = {}".format(compute_emd(d_mat, f_mat)))
