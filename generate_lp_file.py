import math
import numpy as np
import sys, getopt
import matplotlib.pyplot as plt

FILENAME_MODEL = "model.lp"
FILENAME_D = "d.npy"

def generate_points(m: int,
                    n: int,
                    dim: int):
    '''
    this function generates random coordinates and weights for the points
    :param m: size of the first set of points P
    :param n: size of the second set of points Q
    :param dim: number of coordinates for each points
    :return: p_list, wp, q_list, wq
    '''

    p_list = []
    wp = []
    q_list = []
    wq = []
    for i in range(m):
        p_list.append(np.random.normal(loc=1, size=dim))
        wp.append(np.random.uniform(low=0.0, high=1.0))

    for j in range(n):
        q_list.append(np.random.normal(size=dim))
        wq.append(np.random.uniform(low=0.0, high=1.0))

    return p_list, wp, q_list, wq


def compute_d_mat(p_list: list[np.ndarray],
                  q_list: list[np.ndarray]) -> np.ndarray:
    '''
    this function computes the matrix of the distances on all possible edges from P to Q
    :param p_list: list of points in P
    :param q_list: list of points in Q
    :return: d_mat
    '''
    d_mat = np.zeros((len(p_list), len(q_list)))
    for i, p in enumerate(p_list):
        for j, q in enumerate(q_list):
            d_mat[i, j] = math.dist(p, q)
    return d_mat


def generate_lp(p_list: list[np.ndarray],
                wp: list[float],
                q_list: list[np.ndarray],
                wq: list[float],
                d_mat: np.ndarray):
    '''
    Generates the string used for de .lp file
    :param p_list: list of points in P
    :param wp: list of weights for P
    :param q_list: list of points in Q
    :param wq: list of weights for Q
    :param d_mat: matrix of distances across all edges from P to Q
    :return: file_str
    '''
    m = len(p_list)
    n = len(q_list)
    file_str = ""

    # objective function
    obj = "min : "
    for i, p in enumerate(p_list):
        for j, q in enumerate(q_list):
            if i != 0 or j != 0:
                obj += " + "
            obj += "{dij} x{i}.{j}".format(dij=d_mat[i, j], i=i, j=j)
    obj += ";\n"
    file_str += obj

    # first set of constraints
    for i, p in enumerate(p_list):
        con_i = ""
        for j, q in enumerate(p_list):
            if j != 0:
                con_i += " + "
            con_i += "x{i}.{j}".format(i=i, j=j)
        con_i += " <= {wi};\n".format(wi=wp[i])
        file_str += con_i

    # second set of constraints
    for j, q in enumerate(q_list):
        con_j = ""
        for i, p in enumerate(p_list):
            if i != 0:
                con_j += " + "
            con_j += "x{i}.{j}".format(i=i, j=j)
        con_j += " <= {wj};\n".format(wj=wq[j])
        file_str += con_j

    # last constraint
    con_last = ""
    sum_wp = 0
    sum_wq = 0
    for i, p in enumerate(p_list):
        sum_wp += wp[i]

    for j, q in enumerate(q_list):
        sum_wq += wq[j]

    for i, p in enumerate(p_list):
        for j, q in enumerate(q_list):
            if i != 0 or j != 0:
                con_last += " + "
            con_last += "x{i}.{j}".format(i=i, j=j)
    con_last += " = {};\n".format(min(sum_wp, sum_wq))
    file_str += con_last

    return file_str


def write_results(model: str,
                  filename_model: str,
                  d: np.ndarray,
                  filename_d: str):
    '''
    writes the model string to a .lp file and the distance matrix to a .npy file
    we will use the .lp file to solve the linear problem in LPsolve IDE, then export the results file to .csv
    :param model: string following the format for .lp file
    :param filename_model: name of the file to save the model
    :param d: matrix of distances across edges from P to Q
    :param filename_d: name of file to save d
    :return:
    '''
    f = open(filename_model, "w")
    f.write(model)
    f.close()
    np.save(filename_d, d)
    return 0


def plot_weighted_points(p_list: list[np.ndarray],
                         wp: list[float],
                         q_list: list[np.ndarray],
                         wq: list[float],
                         size_factor=1):
    plt.figure()
    plt.title("weighted points", fontsize='small')
    xp = np.array(p_list)
    plt.scatter(xp[:, 0], xp[:, 1], marker='o', c='b', s=200*size_factor*np.array(wp))
    xq = np.array(q_list)
    plt.scatter(xq[:, 0], xq[:, 1], marker='o', c='r', s=200*size_factor*np.array(wq))
    plt.show()


if __name__ == '__main__':
    # p = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([1.0, 1.0])]
    # wp = [0.2, 0.5, 0.3]
    # q = [np.array([0.25, 0.75]), np.array([0.5, -0.5])]
    # wq = [0.4, 0.6]

    # Variables m, n and dim can be used to change the size of
    # the points set and the number of coordinates for each points
    m = 10
    n = 20
    dim = 3
    random_state = 5
    rd_seed = False
    do_plot = False

    # We parse the arguments for script execution in the console
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "m:n:d:r:f:c:p", ["m=", "n=", "dim=", "random_state=",
                                                           "filename_lp=", "filename_d=", "plot="])
    except getopt.GetoptError:
        print(argv)
        print('argument parsing error')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-m', '--m'):
            m = int(arg)
        elif opt in ('-n', '--n'):
            n = int(arg)
        elif opt in ('-d', '--dim'):
            dim = int(arg)
        elif opt in ('-r', '--random_state'):
            if arg == 'N':
                rd_seed = True
            else:
                random_state = int(arg)
        elif opt in ('-f', '--filename_lp'):
            FILENAME_MODEL = arg
        elif opt in ('-c', '--filename_d'):
            FILENAME_D = arg
        elif opt in ('-p', '--plot'):
            do_plot = True

    if not rd_seed:
        np.random.RandomState(seed=random_state)

    p_list, wp, q_list, wq = generate_points(n, m, dim)
    print("{m} + {n} random points generated".format(m=m, n=n))

    d_mat = compute_d_mat(p_list, q_list)
    model = generate_lp(p_list, wp, q_list, wq, d_mat)
    if do_plot:
        plot_weighted_points(p_list, wp, q_list, wq)
    print("lp file created")
    write_results(model, FILENAME_MODEL, d_mat, FILENAME_D)
    print("done, .lp file saved to {fm} and distance matrix saved to {fd}".format(fm=FILENAME_MODEL,
                                                                                  fd=FILENAME_D))

