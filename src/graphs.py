import matplotlib.pyplot as plt


def one_graph(x_values, y_values, x_label=None, y_label=None, textsize=16):
    fig = plt.figure()
    plt.plot(x_values, y_values)
    plt.xlabel(x_label, fontsize=textsize)
    plt.ylabel(y_label, fontsize=textsize)
    return fig


def multiple_models_graph(x_values, y_values, legend=None, x_label=None, y_label=None, textsize=16):
    fig = plt.figure()
    if len(x_values) > len(y_values):
        for y, label in zip(y_values, legend):
            plt.plot(x_values, y, label=label)
    else:
        for x, y, label in zip(x_values, y_values, legend):
            plt.plot(x, y, label=label)
    plt.xlabel(x_label, fontsize=textsize)
    plt.ylabel(y_label, fontsize=textsize)
    plt.legend()
    return fig


def multiple_axes(x_vals_list, y_vals_list, x_labels=None, y_labels=None, legend=None):
    pass
