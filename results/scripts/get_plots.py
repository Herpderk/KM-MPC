import compress_pickle
from os import listdir
from os.path import isfile, join
from typing import List, Tuple
import types

import matplotlib.axes
import matplotlib.axis
import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Times New Roman"})

from km_mpc.experiments.data import *

from consts.trials import SOLVERS, DT
from consts.plots import *
from consts.paths import *


def get_datasets(
    data_paths: List[str]
) -> List[List[TrialData]]:
    datasets_per_solver = []

    for data_path in data_paths:
        datasets_per_traj = []
        file_paths = [
            join(data_path, f)
            for f in listdir(data_path) if isfile(join(data_path, f))
        ]

        for file_path in file_paths:
            with open(file_path, 'rb') as file:
                # Load list of TrialData objects
                try:
                    datasets_per_traj += [compress_pickle.load(
                        file, compression="lzma", set_default_extension=False
                    )]
                except EOFError:
                    continue

        datasets_per_solver += [datasets_per_traj]
        print('Finished loading data from', data_path)
    return datasets_per_solver


def get_solve_time_plot(
    plot_axis: matplotlib.axes.Axes,
    datasets_per_solver: List[List[List[TrialData]]],
    set_xlabel: bool,
    model_label: str = None,
) -> None:
    solve_times_per_solver = []
    for i, dataset in enumerate(datasets_per_solver):
        datasets_per_traj_combined = [
            data
            for data_per_timestep in dataset
            for data in data_per_timestep
            if not is_none(data.mhpe_solution) or not is_none(data.mhpe_solver_stats)
        ]
        if len(datasets_per_traj_combined) == 0:
            continue
        else:
            mhpe_solve_times = get_mhpe_solve_times(datasets_per_traj_combined)
            solve_times_per_solver += [mhpe_solve_times]
            print(f'\n{i}-th solve time min for model {model_label}: {mhpe_solve_times.min()}')
            print(f'{i}-th solve time max for model {model_label}: {mhpe_solve_times.max()}')
            print(f'{i}-th solve time mean for model {model_label}: {mhpe_solve_times.mean()}\n')

            non_convergence_rate = get_mhpe_non_convergence_rate(datasets_per_traj_combined)
            print(f'{i}-th solver non-convergence rate for model {model_label}: {non_convergence_rate}\n')

    vp = plot_axis.violinplot(
        solve_times_per_solver, showmedians=False, showmeans=True, vert=False)
    for part in ('cbars', 'cmeans', 'cmins', 'cmaxes'):
        vp[part].set_color(COLORS)
    for color_index, body in enumerate(vp['bodies']):
        body.set_edgecolor(COLORS[color_index])
        body.set_facecolor('light' + COLORS[color_index])


    plot_axis.set_yticks(
        range(1, 1 + len(X_TICK_LABELS)), labels=X_TICK_LABELS,
        rotation=90, va='center'
    )
    plot_axis.set_xscale('log')
    if set_xlabel:
        plot_axis.set_xlabel('Solve Time (s)')
    if not is_none(model_label):
        plot_axis.set_ylabel(model_label, fontsize=12, fontweight='bold')
        plot_axis.yaxis.set_label_position('left')


def get_trajectory_cost_plot(
    plot_axis: matplotlib.axes.Axes,
    datasets_per_solver: List[List[List[TrialData]]],
    set_xlabel: bool,
    xlim: Tuple[float, float] = None,
) -> None:
    costs_per_solver = [
        np.array([
            get_cost(data_per_timestep) for data_per_timestep in datasets_per_traj])
        for datasets_per_traj in datasets_per_solver
    ]
    for i, costs in enumerate(costs_per_solver):
        print(f'\n{i}-th traj cost min: {costs.min()}')
        print(f'{i}-th traj cost max: {costs.max()}')
        print(f'{i}-th traj cost mean: {costs.mean()}\n')

    vp = plot_axis.violinplot(
        costs_per_solver, showmedians=False, showmeans=True, vert=False)

    for part in ('cbars', 'cmeans', 'cmins', 'cmaxes'):
        vp[part].set_color(COLORS)
    for color_index, body in enumerate(vp['bodies']):
        body.set_edgecolor(COLORS[color_index])
        body.set_facecolor('light' + COLORS[color_index])

    pad = plt.rcParams["xtick.major.size"] + plt.rcParams["xtick.major.pad"]
    def bottom_offset(self, bboxes, bboxes2):
        bottom = self.axes.bbox.ymin
        self.offsetText.set(va="top", ha="left")
        oy = bottom - pad * self.figure.dpi / 72.0
        self.offsetText.set_position((1, oy))

    plot_axis.xaxis._update_offset_text_position = types.MethodType(
        bottom_offset, plot_axis.xaxis)

    if set_xlabel:
        plot_axis.set_xlabel('Trajectory Cost')
    if not is_none(xlim):
        plot_axis.set_xlim(xlim)


def plot_trajectory(
    plot_axis: matplotlib.axis,
    dt: float,
    xs: VectorList,
) -> None:
    interp_N = 1000
    xs_arr = xs.as_array()
    t = dt * np.arange(xs_arr.shape[0])
    plot_axis.set_ylabel('Position Error Norm')
    for i in range(xs_arr.shape[1]):
        t_interp = get_interpolation(t, t, interp_N)
        err_norm = np.linalg.norm(xs_arr[:,:3], axis=1)
        x_interp = get_interpolation(t, err_norm, interp_N)
        color = MONTE_CARLO_COLOR
        plot_axis.plot(t_interp, x_interp, linewidth=2.0, alpha=0.002, color=color)


def get_interpolation(
    Xs: np.ndarray,
    Ys: np.ndarray,
    N: int,
) -> np.ndarray:
    spline_func = make_interp_spline(Xs, Ys)
    interp_x = np.linspace(Xs.min(), Xs.max(), N)
    interp_y = spline_func(interp_x)
    return interp_y


def get_monte_carlo_time_series_plot(
    datasets_per_solver: List[List[List[TrialData]]],
    save_fig: bool,
) -> None:
    for i, datasets_per_traj in enumerate(datasets_per_solver):
        fig, ax = plt.subplots()
        trajectories = [
            get_states(datasets_per_timestep)
            for datasets_per_timestep in datasets_per_traj
        ]
        for trajectory in trajectories:
            plot_trajectory(ax, DT, trajectory)

        ax.set(xlabel='Time (s)')
        ax.set_yscale('log')
        if save_fig:
            fig.savefig(FIG_PATH + f'/monte_carlo_{i}.png', dpi=400)
        #ax.text(0.75, 0.75, f'n trials: {len(trajectories)}', fontsize=12)


def get_plots_per_model(
    model_label: str,
    solve_time_axis: matplotlib.axes.Axes,
    traj_cost_axis: matplotlib.axes.Axes,
    traj_cost_crop_axis: matplotlib.axes.Axes,
    traj_cost_max: float,
    data_path: str,
    set_xlabels: bool,
    save_monte_carlo=False,
) -> None:
    data_paths = [data_path + plugin + '/' for plugin in SOLVERS.keys()]
    datasets_per_solver = get_datasets(data_paths)
    get_solve_time_plot(
        solve_time_axis, datasets_per_solver, set_xlabels, model_label)
    get_trajectory_cost_plot(
        traj_cost_axis, datasets_per_solver, set_xlabels)
    get_trajectory_cost_plot(
        traj_cost_crop_axis, datasets_per_solver, set_xlabels, traj_cost_max)
    get_monte_carlo_time_series_plot(datasets_per_solver, save_monte_carlo)


def main():
    fig, axs = plt.subplots(2, 3, sharey='row')
    fig.align_xlabels()
    for ax in axs.reshape(-1):
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    # Add Crazyflie plots
    data_path = DATA_PATH + CRAZYFLIE_PATH
    get_plots_per_model(
        'Crazyflie', axs[0,0], axs[0,1], axs[0,2], [-2.5e4, 5.0e5], data_path, False)

    # Add Fusion One plots
    data_path = DATA_PATH + FUSION_ONE_PATH
    get_plots_per_model(
        'Fusion 1', axs[1,0], axs[1,1], axs[1,2], [-1.5e5, 3.0e6], data_path, True, True)

    fig.savefig(FIG_PATH + '/violin.png', dpi=400)
    plt.show()


if __name__=='__main__':
    main()
