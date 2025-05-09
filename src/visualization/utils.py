from typing import Any

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Rectangle
from matplotlib.pyplot import Axes


# shameless theft from https://stackoverflow.com/a/55501861, i.e.
# https://stackoverflow.com/questions/55501860/how-to-put-multiple-colormap-patches-in-a-matplotlib-legend
class HandlerColormap(HandlerBase):
    def __init__(self, cmap: plt.cm, num_stripes: int = 8, **kwargs: Any):
        HandlerBase.__init__(self, **kwargs)
        self.cmap = cmap
        self.num_stripes = num_stripes

    def create_artists(  # type: ignore
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        stripes = []
        for i in range(self.num_stripes):
            s = Rectangle(
                [xdescent + i * width / self.num_stripes, ydescent],
                width / self.num_stripes,
                height,
                fc=self.cmap((2 * i + 1) / (2 * self.num_stripes)),
                transform=trans,
                edgecolor="none",
            )
            stripes.append(s)
        return stripes


def add_gradient_patch_legend(
    fig: plt.Figure, ax: Axes, cmap: plt.cm, gradient_label: str, **legend_kwargs: Any
) -> None:
    existing_handles, existing_labels = ax.get_legend_handles_labels()
    cmap_handles = [Rectangle((0, 0), 1, 1)]
    # seems you only need a handler map for legend element that need a custom handler
    handler_map = dict(zip(cmap_handles, [HandlerColormap(cmap, num_stripes=128)]))
    labels = [*existing_labels, gradient_label]
    handles = [*existing_handles, *cmap_handles]
    fig.legend(
        handles=handles,
        labels=labels,
        handler_map=handler_map,
        **legend_kwargs,
    ).set_visible(True)
