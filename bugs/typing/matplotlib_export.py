import matplotlib.pyplot as plt


def make_plot() -> tuple[plt.Figure, plt.Axes]:  # ❌ reportPrivateImportUsage
    fig, axes = plt.subplots()
    # plotting code goes here
    return fig, axes
