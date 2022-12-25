import matplotlib.pyplot as plt


def one_graph(x_values, y_values, x_label=None, y_label=None, textsize=16):
    fig = plt.figure()
    plt.plot(x_values, y_values)
    plt.xlabel(x_label, fontsize=textsize)
    plt.ylabel(y_label, fontsize=textsize)
    return fig


def multiple_models_one_x(x_values, y_values, legend=None, x_label=None, y_label=None, textsize=16):
    fig = plt.figure()
    for y, label in zip(y_values, legend):
        plt.plot(x_values, y, label=label)
    plt.xlabel(x_label, fontsize=textsize)
    plt.ylabel(y_label, fontsize=textsize)
    plt.legend()
    return fig


def multiple_models_multi_x(x_values, y_values, legend=None, x_label=None, y_label=None, textsize=16):
    fig = plt.figure()
    for x, y, label in zip(x_values, y_values, legend):
        plt.plot(x, y, label=label)
    plt.xlabel(x_label, fontsize=textsize)
    plt.ylabel(y_label, fontsize=textsize)
    plt.legend()
    return fig


def scatter_plots(x_values, y_values, textsize=18):
    fig = plt.figure()
    ax0 = fig.add_axes([0, 1, 1, 1])
    ax1 = fig.add_axes([1.1, 1, 1, 1])
    ax2 = fig.add_axes([0, 0, 1, 1])
    ax3 = fig.add_axes([1.1, 0, 1, 1])
    ax0.scatter(x_values[0], y_values[0])
    ax0.set_ylabel('Wartości przewidziane', fontsize=textsize)
    ax0.set_xlabel('Wartości docelowe', fontsize=textsize)
    ax0.legend(["MultiLayerANN"], fontsize=textsize)

    ax1.scatter(x_values[1], y_values[1], c='orange')
    ax1.set_xlabel('Wartości docelowe', fontsize=textsize)
    ax1.legend(["LeakySNN"], fontsize=textsize)

    ax2.scatter(x_values[2], y_values[2], c='green')
    ax2.set_ylabel('Wartości przewidziane', fontsize=textsize)
    ax2.set_xlabel('Wartości docelowe', fontsize=textsize)
    ax2.legend(["SynapticSNN"], fontsize=textsize)

    ax3.scatter(x_values[3], y_values[3], c='red')
    ax3.set_xlabel('Wartości docelowe', fontsize=textsize)
    ax3.legend(["DoubleLeakySNN"], fontsize=textsize)
