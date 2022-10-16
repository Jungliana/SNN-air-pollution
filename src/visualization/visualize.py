from matplotlib import pyplot as plt


def dummy_plot(data):
    _, ax = plt.subplots()
    ax.plot(data["x"], data["y"])
    return ax
