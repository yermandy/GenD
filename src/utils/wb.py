import numpy as np
import pandas as pd
import wandb

from .decorators import TryExcept


@TryExcept()
def create_custom_wandb_metric(
    xs: list,
    ys: list,
    classes: list,
    title: str = "Precision Recall Curve",
    x_axis_title: str = "Recall",
    y_axis_title: str = "Precision",
):
    """Creates a custom wandb metric similar to default wandb.plot.pr_curve

    Args:
        xs: list of N values to plot on the x-axis
        ys: list of N values to plot on the y-axis
        classes: class labels for each point (list of N values)
        title: plot title

    Returns:
        wandb object to log
    """
    df = pd.DataFrame(
        {
            "class": classes,
            "y": ys,
            "x": xs,
        }
    ).round(3)

    return wandb.plot_table(
        "wandb/area-under-curve/v0",
        wandb.Table(dataframe=df),
        {"x": "x", "y": "y", "class": "class"},
        {
            "title": title,
            "x-axis-title": x_axis_title,
            "y-axis-title": y_axis_title,
        },
    )


@TryExcept()
def plot_curve_wandb(
    xs: np.ndarray,
    ys: np.ndarray,
    names: list = [],
    id: str = "precision-recall",
    title: str = "Precision Recall Curve",
    x_axis_title: str = "Recall",
    y_axis_title: str = "Precision",
    num_xs: int = 100,
    only_mean: bool = True,
):
    """adds a metric curve to wandb

    Args:
        xs: np.array of N values
        ys: np.array of C by N values where C is the number of classes
        names: dict of class names
        id: log id in wandb
        title: plot title in wandb
        num_xs: number of points to interpolate to
        only_mean: if True, only the mean curve is plotted
    """
    # create new xs
    xs_new = np.linspace(xs[0], xs[-1], num_xs)

    # create arrays for logging
    xs_log = xs_new.tolist()
    ys_log = np.interp(xs_new, xs, np.mean(ys, axis=0)).tolist()
    classes = ["mean"] * len(xs_log)

    if not only_mean and len(names) == len(ys):
        for i, y in enumerate(ys):
            # add new xs
            xs_log.extend(xs_new)
            # interpolate y to new xs
            ys_log.extend(np.interp(xs_new, xs, y))
            # add class names
            classes.extend([names[i]] * len(xs_new))

    wandb.log(
        {
            id: create_custom_wandb_metric(
                xs_log,
                ys_log,
                classes,
                title,
                x_axis_title,
                y_axis_title,
            )
        },
        commit=False,
    )
