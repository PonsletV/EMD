from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import generate_lp_file as gen
import sys, getopt


def build_dataset(n_samples: int,
                  dim: int,
                  centers,
                  std=1.0,
                  random_state=None):
    '''
    generates clustered data for k means application
    :param n_samples:
    :param dim:
    :param centers:
    :param std:
    :param random_state:
    :return:
    '''
    df = make_blobs(n_samples=n_samples,
                    n_features=dim,
                    centers=centers,
                    random_state=random_state,
                    cluster_std=std)
    return df


def plot_clusters(df):
    '''
    plots the clusters
    :param df: a tuple in the form (points, label)
    :return:
    '''
    plt.figure()
    plt.title("clusters", fontsize='small')
    X1, Y1 = df
    plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1,
                s=25, edgecolor='k')
    plt.show()


def kmeans(X,
           n_clusters: int):
    '''
    K means algorithm
    :param X:
    :param n_clusters:
    :return:
    '''
    clust = KMeans(n_clusters).fit(X)
    return clust


def clusters_to_weighted_pts(X, clusters):
    '''
    Tranforms a clustering to a list of points and a list of associated weights
    :param X: All the points in the clustered data
    :param clusters: fitted kmeans object
    :return:
    '''
    p = list(clusters.cluster_centers_)
    weights = [0]*len(p)
    for l in clusters.labels_:
        weights[l] += 1/len(X)
    return p, weights


if __name__ == '__main__':
    random_state = 7

    n_samples = 1000
    dim = 2
    n_centers = 6
    stddev = 1.5
    FILENAME_MODEL = gen.FILENAME_MODEL
    FILENAME_D = gen.FILENAME_D
    n_clusters_1 = 10
    n_clusters_2 = 8
    do_plot = False

    # We parse the arguments for script execution in the console
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "n:r:f:c:p", ["n_samples=", "dim=", "random_state=",
                                                       "filename_lp=", "filename_d=", "plot="])
    except getopt.GetoptError:
        print(argv)
        print('argument parsing error')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-n', '--n_samples'):
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

    df = build_dataset(n_samples, dim, n_centers, stddev, random_state)

    (X, _) = df

    clusters1 = kmeans(X, n_clusters_1)

    (p, wp) = clusters_to_weighted_pts(X, clusters1)

    clusters2 = kmeans(X, n_clusters_2)

    (q, wq) = clusters_to_weighted_pts(X, clusters2)

    if do_plot:
        plot_clusters(df)
        plot_clusters((X, clusters1.labels_))
        plot_clusters((X, clusters2.labels_))
        gen.plot_weighted_points(p, wp, q, wq)

    d_mat = gen.compute_d_mat(p, q)
    model = gen.generate_lp(p, wp, q, wq, d_mat)

    print("lp file created")
    gen.write_results(model, FILENAME_MODEL, d_mat, FILENAME_D)



