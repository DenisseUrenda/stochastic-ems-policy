# source/plotting.py

import numpy as np
import matplotlib.pyplot as plt


def plot_arrivals_departures(
    t_hist,
    A_hist,
    B_hist,
    dt=0.5,
    n_trajectories=3,
    save_path=None
):
    fig, ax = plt.subplots(2, 1, sharex=True)

    n_plot = min(n_trajectories, A_hist.shape[1])

    for tr in range(n_plot):

        ax[0].bar(
            t_hist.detach().cpu().numpy()[1:],
            A_hist[:, tr].detach().cpu().numpy(),
            width=dt * 0.9,
            alpha=0.5,
            label=f"Trajectory {tr+1}"
        )

        ax[1].bar(
            t_hist.detach().cpu().numpy()[1:],
            B_hist[:, tr].detach().cpu().numpy(),
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



def plot_state_trajectories(t_hist, S_hist, save_path=None):

    N_hist = S_hist[..., 0]
    D_hist = S_hist[..., 1]
    E_hist = S_hist[..., 2]
    P_hist = S_hist[..., 3]

    fig, ax = plt.subplots(4, 1, sharex=True)

    ax[0].plot(t_hist.detach().cpu().numpy(), N_hist.detach().cpu().numpy(), c="0.8")
    ax[0].text( 0.01, 0.8, r"No. Parked EVs $N_t$", transform=ax[0].transAxes)

    ax[1].plot(t_hist.detach().cpu().numpy(), D_hist.detach().cpu().numpy(), c="0.8")
    ax[1].text( 0.01, 0.8, r"Pending Demand $D_t$ (kWh)", transform=ax[1].transAxes)

    ax[2].plot(t_hist.detach().cpu().numpy(), E_hist.detach().cpu().numpy(), c="0.8")
    ax[2].text(0.01, 0.8, r"Stored Energy $E_t$ (kWh)", transform=ax[2].transAxes)

    ax[3].plot(t_hist.detach().cpu().numpy(), P_hist.detach().cpu().numpy(), c="0.8")
    ax[3].text(0.01, 0.8, r"Energy Price $P_t$ (USD/kWh)", transform=ax[3].transAxes)

    ax[3].set_xticks(np.arange(0, 24, 4))
    ax[3].set_xlabel(r"Time $t$ (hrs)")

    plt.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.2)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def plot_endogenous_trajectories(t_hist, endog_hist, save_path=None):

    q_hist  = endog_hist[..., 0]
    ec_hist = endog_hist[..., 1]
    ed_hist = endog_hist[..., 2]
    Ub_hist = endog_hist[..., 3]
    Us_hist = endog_hist[..., 4]

    fig, ax = plt.subplots(5, 1, sharex=True, figsize=(7,5))

    ax[0].plot(t_hist.detach().cpu().numpy(), q_hist.detach().cpu().numpy(), c='0.8')
    ax[0].text(0.01, 0.7, r'Power allocated $q_t$ (kW)', transform=ax[0].transAxes)

    ax[1].plot(t_hist.detach().cpu().numpy(), ec_hist.detach().cpu().numpy(), c='0.8')
    ax[1].text(0.01, 0.7, r'Charged energy $e_t^c$ (kWh)', transform=ax[1].transAxes)

    ax[2].plot(t_hist.detach().cpu().numpy(), ed_hist.detach().cpu().numpy(), c='0.8')
    ax[2].text(0.01, 0.7, r'Energy discharged $e_t^d$ (kWh)', transform=ax[2].transAxes)

    ax[3].plot(t_hist.detach().cpu().numpy(), Ub_hist.detach().cpu().numpy(), c='0.8')
    ax[3].text(0.01, 0.7, r'Energy purchased $U_t^b$ (kWh)', transform=ax[3].transAxes)

    ax[4].plot(t_hist.detach().cpu().numpy(), Us_hist.detach().cpu().numpy(), c='0.8')
    ax[4].text(0.01, 0.7, r'Energy sold $U_t^s$ (kWh)', transform=ax[4].transAxes)

    ax[4].set_xticks(np.arange(0, 24, 4))
    ax[4].set_xlabel(r'Time $t$ (hrs)')

    plt.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.2)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()



def plot_control_trajectories(t_hist, u_hist, save_path=None):

    eta_hist   = u_hist[..., 0]
    delta_hist = u_hist[..., 1]
    gamma_hist = u_hist[..., 2]

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(7,5))

    ax[0].plot(t_hist.detach().cpu().numpy()[:-1], eta_hist.detach().cpu().numpy(), c='0.8')
    ax[0].text(0.01, 0.8, r'Power allocated proportion $\eta_t \in [0,1]$', transform=ax[0].transAxes)

    ax[1].axhline(0, linestyle='--', c='0.0')
    ax[1].plot(t_hist.detach().cpu().numpy()[:-1], delta_hist.detach().cpu().numpy(), c='0.8')
    ax[1].text(0.01, 0.8, r'Charge/discharge energy proportion $\delta_t \in [-1,1]$', transform=ax[1].transAxes)

    ax[2].plot(t_hist.detach().cpu().numpy()[:-1], gamma_hist.detach().cpu().numpy(), c='0.8')
    ax[2].text(0.01, 0.8, r'Energy purchase proportion $\gamma_t \in [0,1]$', transform=ax[2].transAxes)

    ax[2].set_xticks(np.arange(0,24,4))
    ax[2].set_xlabel(r'Time $t$ (hrs)')

    plt.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.2)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()


def plot_cost_trajectories(t_hist, c_hist, J_episode=None, save_path=None):

    plt.plot(
        t_hist.detach().cpu().numpy()[:-1],
        c_hist.detach().cpu().numpy(),
        c='0.8'
    )

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