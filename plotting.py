from matplotlib.axes import Axes
from matplotlib.ticker import LogFormatter, LogLocator


def log_format(ax: Axes) -> None:
    ax.set_yscale("log")

    class PlainLogFormatter(LogFormatter):
        def __call__(self, x: float, pos: int | None = None, tol: float = 1e-8) -> str:
            label = super().__call__(x, pos)
            if label == "":
                return ""
            try:
                value = float(x)
                if abs(value - int(value)) < tol:
                    return str(int(value))
                else:  # noqa: RET505
                    return f"{x:.4g}"
            except Exception:  # noqa: BLE001
                return label

    # y-axis ticks
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax.yaxis.set_major_formatter(LogFormatter())

    ax.yaxis.set_minor_locator(LogLocator(base=10, subs="auto", numticks=10))
    ax.yaxis.set_minor_formatter(PlainLogFormatter(minor_thresholds=(3, 1)))
