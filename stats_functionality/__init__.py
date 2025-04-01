    # results = analyze_max_stable_process(data, variable_name='tp', block_size=24)
from .max_stable import analyze_max_stable_process, plot_max_stable_results

__all__ = [
    "analyze_max_stable_process",
    "plot_max_stable_results",
]