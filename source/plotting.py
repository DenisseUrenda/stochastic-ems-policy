# source/plotting.py

import numpy as np
import matplotlib.pyplot as plt


def plot_arrivals_departures(
    history,
    dt=0.5,
    n_trajectories=3,
    save_path=None
):
    fig, ax = plt.subplots(2, 1, sharex=True)

    time = history["time"]
    arrival = history["arrival"]
    departure = history["departure"]

    n_plot = min(n_trajectories, arrival.shape[1])

    for tr in range(n_plot):

        ax[0].bar(
            time.detach().cpu().numpy()[1:],
            arrival[:, tr].detach().cpu().numpy(),
            width=dt * 0.9,
            alpha=0.5,
            label=f"Trajectory {tr+1}"
        )

        ax[1].bar(
            time.detach().cpu().numpy()[1:],
            departure[:, tr].detach().cpu().numpy(),
            width=dt * 0.9,
            alpha=0.5,
            label=f"Trajectory {tr+1}"
        )

    ax[0].set_ylabel(r"No. Arrivals $A_t$")
    ax[1].set_ylabel(r"No. Departures $B_t$")

    ax[1].set_xticks(np.arange(0, 24, 4))
    ax[1].set_xlabel(r"Time $t$ (hrs)")

    ax[0].legend(frameon=False)

    plt.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.2)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()



def plot_state_trajectories(history, save_path=None):

    fig, ax = plt.subplots(4, 1, sharex=True)

    time = history["time"]
    state = history["state"]

    t = time.detach().cpu().numpy()

    N = state[..., 0].detach().cpu().numpy()
    D = state[..., 1].detach().cpu().numpy()
    E = state[..., 2].detach().cpu().numpy()
    P = state[..., 3].detach().cpu().numpy()

    ax[0].plot(t, N, c="0.8", lw=0.5)
    ax[0].plot(t, N.mean(axis=1), c="C0", lw=2)
    ax[0].text(0.01, 0.8, r"No. Parked EVs $N_t$", transform=ax[0].transAxes)

    ax[1].plot(t, D, c="0.8", lw=0.5)
    ax[1].plot(t, D.mean(axis=1), c="C0", lw=2)
    ax[1].text(0.01, 0.8, r"Pending Demand $D_t$ (kWh)", transform=ax[1].transAxes)

    ax[2].plot(t, E, c="0.8", lw=0.5)
    ax[2].plot(t, E.mean(axis=1), c="C0", lw=2)
    ax[2].text(0.01, 0.8, r"Stored Energy $E_t$ (kWh)", transform=ax[2].transAxes)

    ax[3].plot(t, P, c="0.8", lw=0.5)
    ax[3].plot(t, P.mean(axis=1), c="C0", lw=2)
    ax[3].text(0.01, 0.8, r"Energy Price $P_t$ (USD/kWh)", transform=ax[3].transAxes)

    plt.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.2)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def plot_endogenous_trajectories(history, save_path=None):

    time = history["time"]
    endog = history["endog"]

    t = time.detach().cpu().numpy()

    q  = endog[..., 0].detach().cpu().numpy()
    ec = endog[..., 1].detach().cpu().numpy()
    ed = endog[..., 2].detach().cpu().numpy()
    Ub = endog[..., 3].detach().cpu().numpy()
    Us = endog[..., 4].detach().cpu().numpy()

    fig, ax = plt.subplots(5, 1, sharex=True, figsize=(7,5))

    ax[0].plot(t, q, c='0.8', lw=0.5)
    ax[0].plot(t, q.mean(axis=1), c='C0', lw=2)
    ax[0].text(0.01, 0.7, r'Power allocated $q_t$ (kW)', transform=ax[0].transAxes)

    ax[1].plot(t, ec, c='0.8', lw=0.5)
    ax[1].plot(t, ec.mean(axis=1), c='C0', lw=2)
    ax[1].text(0.01, 0.7, r'Charged energy $e_t^c$ (kWh)', transform=ax[1].transAxes)

    ax[2].plot(t, ed, c='0.8', lw=0.5)
    ax[2].plot(t, ed.mean(axis=1), c='C0', lw=2)
    ax[2].text(0.01, 0.7, r'Energy discharged $e_t^d$ (kWh)', transform=ax[2].transAxes)

    ax[3].plot(t, Ub, c='0.8', lw=0.5)
    ax[3].plot(t, Ub.mean(axis=1), c='C0', lw=2)
    ax[3].text(0.01, 0.7, r'Energy purchased $U_t^b$ (kWh)', transform=ax[3].transAxes)

    ax[4].plot(t, Us, c='0.8', lw=0.5)
    ax[4].plot(t, Us.mean(axis=1), c='C0', lw=2)
    ax[4].text(0.01, 0.7, r'Energy sold $U_t^s$ (kWh)', transform=ax[4].transAxes)

    ax[4].set_xticks(np.arange(0, 24, 4))
    ax[4].set_xlabel(r'Time $t$ (hrs)')

    plt.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.2)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()



def plot_control_trajectories(history, save_path=None):

    time = history["time"]
    control = history["control"]

    t = time.detach().cpu().numpy()[:-1]

    eta   = control[..., 0].detach().cpu().numpy()
    delta = control[..., 1].detach().cpu().numpy()
    gamma = control[..., 2].detach().cpu().numpy()

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(7,5))

    ax[0].plot(t, eta, c='0.8', lw=0.5)
    ax[0].plot(t, eta.mean(axis=1), c='C0', lw=2)
    ax[0].text(0.01, 0.8, r'Power allocated proportion $\eta_t \in [0,1]$', transform=ax[0].transAxes)

    ax[1].axhline(0, linestyle='--', c='0.0')
    ax[1].plot(t, delta, c='0.8', lw=0.5)
    ax[1].plot(t, delta.mean(axis=1), c='C0', lw=2)
    ax[1].text(0.01, 0.8, r'Charge/discharge energy proportion $\delta_t \in [-1,1]$', transform=ax[1].transAxes)

    ax[2].plot(t, gamma, c='0.8', lw=0.5)
    ax[2].plot(t, gamma.mean(axis=1), c='C0', lw=2)
    ax[2].text(0.01, 0.8, r'Energy purchase proportion $\gamma_t \in [0,1]$', transform=ax[2].transAxes)

    ax[2].set_xticks(np.arange(0,24,4))
    ax[2].set_xlabel(r'Time $t$ (hrs)')

    plt.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.2)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()


def plot_cost_trajectories(history, save_path=None):

    time = history["time"]
    cost = history["cost"]
    J_episode = history["J_episode"]

    t = time.detach().cpu().numpy()[:-1]
    c = cost.detach().cpu().numpy()

    plt.plot(t, c, c='0.8', lw=0.5)
    plt.plot(t, c.mean(axis=1), c='C0', lw=2)

    plt.axhline(0, linestyle='--', c='0.0')

    plt.ylabel(r'Interval cost $c_t$ (USD)')

    if J_episode is not None:
        plt.title(
            f'Mean cumulative cost J: {J_episode:0.2f}',
            loc='right',
            size=14
        )

    plt.xticks(np.arange(0,24,4))
    plt.xlabel(r'Time $t$ (hrs)')

    plt.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.2)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()


def plot_variable_across_policies(
    results,
    key,
    title = None,
    filename = None,
    figsize = None,
    alpha=0.3,
    linewidth=2,
):
    policies = list(results.keys())
    n_policies = len(policies)

    if figsize is None:
        figsize = (5 * n_policies, 6)

    fig, axes = plt.subplots(
        1, n_policies, 
        figsize=figsize,
        sharex=True, sharey=True
    )

    if len(policies) == 1:
        axes = [axes]

    for ax, policy_name in zip(axes, policies):

        hist = results[policy_name]

        t = hist["time"].detach().numpy()

        # ---------- states ----------
        if key == "N": y = hist["state"][..., 0].detach().numpy()
        elif key == "D": y = hist["state"][..., 1].detach().numpy()
        elif key == "E": y = hist["state"][..., 2].detach().numpy()
        elif key == "P": y = hist["state"][..., 3].detach().numpy()

        # ---------- controls ----------
        elif key == "eta": y = hist["control"][..., 0].detach().numpy()
        elif key == "delta": y = hist["control"][..., 1].detach().numpy()
        elif key == "gamma": y = hist["control"][..., 2].detach().numpy()

        # ---------- endogenous ----------
        elif key == "q": y = hist["endog"][..., 0].detach().numpy()
        elif key == "ec": y = hist["endog"][..., 1].detach().numpy()
        elif key == "ed": y = hist["endog"][..., 2].detach().numpy()
        elif key == "Ub": y = hist["endog"][..., 3].detach().numpy()
        elif key == "Us": y = hist["endog"][..., 4].detach().numpy()

        # ---------- events ----------
        elif key == "A": y = hist["arrival"].detach().numpy()
        elif key == "B": y = hist["departure"].detach().numpy()

        # ---------- costs ----------
        elif key == "c": y = hist["cost"].detach().numpy()

        else:
            raise ValueError(f"Unknown key: {key}")

        n = min(len(t), y.shape[0])

        t = t[:n]
        y = y[:n]
        ax.plot(t, y, color="0.8", alpha=alpha, linewidth=1)

        y_mean = y.mean(axis=1)
        ax.plot(t, y_mean, linewidth=linewidth)

        ax.set_title(policy_name.capitalize())

    axes[0].set_ylabel(title)

    for ax in axes:
        ax.set_xlabel(r"Time $t$ (hrs)")

    fig.suptitle(title)

    plt.tight_layout()

    if filename is not None:
        fig.savefig(filename, bbox_inches="tight", dpi=300,)

    plt.show()

    return fig, axes



from itertools import cycle

def plot_policy_metric(results, metric_key="cost", metric_idx=None, ax=None, policies=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    if policies is None:
        policies = list(results.keys())

    colors = cycle([
        "#4C78A8",
        "#F28E2B",
        "#59A14F",
        "#B07AA1",
        "#76B7B2",
        "#E15759",
        "#9C755F",
        "#BAB0AC",
    ])

    markers = cycle(["o", "s", "^", "D", "v", "P", "X", "*"])

    reference_policy = policies[0]
    time = results[reference_policy]["time"].detach().numpy()

    handles = []
    labels = []

    for policy_name, color, marker in zip(policies, colors, markers):
        history = results[policy_name]
        y = history[metric_key].detach().numpy()

        if metric_idx is not None:
            y = y[..., metric_idx]

        y_mean = y.mean(axis=1)

        if len(y_mean) == len(time):
            x = time
        elif len(y_mean) == len(time) - 1:
            x = time[:-1]
        else:
            raise ValueError(
                f"Time length {len(time)} is not compatible with {metric_key} length {len(y_mean)}."
            )

        line = ax.plot(
            x,
            y_mean,
            color=color,
            lw=1.5,
            alpha=0.8,
            marker=marker,
            ms=4,
            markevery=3,
            label=policy_name,
        )

        handles += line
        labels.append(policy_name)

    ax.set_xlabel("Time (hour)")
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, alpha=0.10)

    return ax, handles, labels




def plot_controls_summary(results, save_path=None):

    fig, ax = plt.subplots(2, 3, figsize=(13, 8), sharex=True)
    ax = ax.flatten()

    plot_specs = [
        ("eta", r"Control $\bar{\eta}_t$", r""),
        ("delta", r"Control $\bar{\delta}_t$", r""),
        ("gamma", r"Control $\bar{\gamma}_t$", r""),
        ("q", r"EV power allocation $\bar{q}_t$", r"kW"),
        ("E", r"Stored energy $\bar{E}_t$", r"kWh"),
        ("Ub", r"Purchased energy $\bar{U}^{b}_t$", r"kWh"),
    ]

    linestyles = [
        "-",
        "--",
        "-.",
        ":",
        (0, (5, 1)),
        (0, (3, 1, 1, 1)),
    ]

    def get_series(hist, key):
        t = hist["time"].detach().numpy()

        if key == "E":
            y = hist["state"][..., 2].detach().numpy()
        elif key == "eta":
            y = hist["control"][..., 0].detach().numpy(); t = t[:-1]
        elif key == "delta":
            y = hist["control"][..., 1].detach().numpy(); t = t[:-1]
        elif key == "gamma":
            y = hist["control"][..., 2].detach().numpy(); t = t[:-1]
        elif key == "q":
            y = hist["endog"][..., 0].detach().numpy(); t = t[:-1]
        elif key == "Ub":
            y = hist["endog"][..., 3].detach().numpy(); t = t[:-1]
        else:
            raise ValueError(f"Unknown key: {key}")

        y_mean = y.mean(axis=1)
        n = min(len(t), len(y_mean))
        return t[:n], y_mean[:n]

    for j, (key, title, ylabel) in enumerate(plot_specs):

        for i, (policy_name, hist) in enumerate(results.items()):
            t, y_mean = get_series(hist, key)
            ax[j].plot(
                t, y_mean, 
                linestyle=linestyles[i % len(linestyles)], 
                linewidth=2, 
                label=policy_name.capitalize()
            )

        ax[j].text(
            0.95, 0.95, 
            title, transform=ax[j].transAxes, fontsize=13, 
            ha="right", va="top", 
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
        )

        if ylabel:
            ax[j].set_ylabel(ylabel)

    for a in ax[-3:]:
        a.set_xlabel(r"Time $t$ (hrs)")

    handles, labels = ax[0].get_legend_handles_labels()
    legend = fig.legend(
        handles, labels, loc="upper center", 
        ncol=len(results), bbox_to_anchor=(0.5, 1.05), 
        frameon=False
    )

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", bbox_extra_artists=[legend])

    plt.show()

    return fig, ax





def plot_variable_across_lambdas(traj_results, key, title, filename, alpha=0.5, linewidth=2):
    lambdas = list(traj_results.keys())

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, lambda_cost in zip(axes, lambdas):
        res = traj_results[lambda_cost]

        t = res["time"].numpy()

        if key == "N": y = res["state"][..., 0].numpy()
        elif key == "D": y = res["state"][..., 1].numpy()
        elif key == "E": y = res["state"][..., 2].numpy()
        elif key == "P": y = res["state"][..., 3].numpy()

        elif key == "eta": y = res["control"][..., 0].numpy()
        elif key == "delta": y = res["control"][..., 1].numpy()
        elif key == "gamma": y = res["control"][..., 2].numpy()

        elif key == "q": y = res["endog"][..., 0].numpy()
        elif key == "ec": y = res["endog"][..., 1].numpy()
        elif key == "ed": y = res["endog"][..., 2].numpy()
        elif key == "Ub": y = res["endog"][..., 3].numpy()
        elif key == "Us": y = res["endog"][..., 4].numpy()

        elif key == "A": y = res["arrival"].numpy()
        elif key == "B": y = res["departure"].numpy()

        elif key == "c": y = res["cost"].numpy()
        else:
            raise ValueError(f"Unknown key: {key}")

        n = min(len(t), y.shape[0])
        t = t[:n]
        y = y[:n, :]

        ax.plot(t, y, c="0.8", alpha=alpha, linewidth=linewidth)

        y_mean = y.mean(axis=1)
        ax.plot(t, y_mean, linewidth=linewidth)

        ax.text(
            0.04, 0.9,
            rf"$\lambda = {lambda_cost}$",
            transform=ax.transAxes,
            fontsize=15,
            weight="bold"
        )


    for ax in axes[-2:]:
        ax.set_xlabel(r"Time $t$ (hrs)")

    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    plt.show()

    return fig, ax


def plot_mean_trajectory_summary(traj_results, untrained_res, save_path=None):

    fig, ax = plt.subplots(2, 3, figsize=(13, 8), sharex=True)
    ax = ax.flatten()

    plot_specs = [
        ("eta", r"Control $\bar{\eta}_t$", r""),
        ("delta", r"Control $\bar{\delta}_t$", r""),
        ("gamma", r"Control $\bar{\gamma}_t$", r""),
        ("q", r"EV power allocation $\bar{q}_t$", r"kW"),
        ("E", r"Stored energy $\bar{E}_t$", r"kWh"),
        ("Ub", r"Purchased energy $\bar{U}^{b}_t$", r"kWh"),
    ]

    linestyles = ["-", "--", "-.", ":"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    def get_series(res, key):
        t = res["time"].numpy()

        if key == "E":
            y = res["state"][..., 2].numpy()
        elif key == "eta":
            y = res["control"][..., 0].numpy(); t = t[:-1]
        elif key == "delta":
            y = res["control"][..., 1].numpy(); t = t[:-1]
        elif key == "gamma":
            y = res["control"][..., 2].numpy(); t = t[:-1]
        elif key == "q":
            y = res["endog"][..., 0].numpy(); t = t[:-1]
        elif key == "Ub":
            y = res["endog"][..., 3].numpy(); t = t[:-1]
        else:
            raise ValueError(f"Unknown key: {key}")

        y_mean = y.mean(axis=1)
        n = min(len(t), len(y_mean))
        return t[:n], y_mean[:n]

    for j, (key, title, ylabel) in enumerate(plot_specs):

        t, y_mean = get_series(untrained_res, key)
        ax[j].plot(t, y_mean, linestyle=(0, (3, 1, 1, 1)), color="0.35", linewidth=2, label="Untrained")

        for i, (lambda_cost, res) in enumerate(traj_results.items()):
            t, y_mean = get_series(res, key)
            ax[j].plot(t, y_mean, linestyle=linestyles[i], color=colors[i], linewidth=2, label=rf"$\lambda={lambda_cost}$")

        ax[j].text(0.95, 0.95, title, transform=ax[j].transAxes, fontsize=13, ha="right", va="top", bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

        if ylabel:
            ax[j].set_ylabel(ylabel)

    for a in ax[-3:]:
        a.set_xlabel(r"Time $t$ (hrs)")

    handles, labels = ax[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.05), frameon=False)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", bbox_extra_artists=[legend])

    plt.show()