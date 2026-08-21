import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class AlleePopulation:
    """
    The population-with-Allee-effects SDE:

        dn = mu(n; <n>) dt + sigma * sqrt(n) dW
        mu = a1 n + a2 n^2 + a3 n^3 + a4,
        a1(0) = -r + phi1 S <n>
        a2(0) = r / T + r / Kc - phi1 - phi2 S <n>

    old_params: the initial effective and target forces (pre-perturbation)
    new_params: the equilibrium target forces (post-perturbation)
    """

    def __init__(self, old_params, new_params, sigma):
        # --- inputs --------------------------------------------------------------
        self.old_params = old_params      # dict: r, T, Kc, m, phi1, phi2, S
        self.new_params = new_params      # dict: same keys, perturbed values
        self.sigma = sigma

        # --- histories: one record (dict) per saved time step --------------------------------------
        self.dme_history = []             # keys: time, effective_forces, target_forces, expectations
        self.dymes_history = []           # keys: time, effective_forces, expectations
        self.stochastic_history = []      # keys: time, expectations
        self.full_stochastic_distributions = [] # to save the full stochastic distributions at a few time points

        # --- current state at t = 0 --------------------------------------------
        self.time = 0.0
        self.avN = self.equilibrium_avN()  # current mean abundance <n>
        self.effective_forces = self.forces(old_params)
        self.target_forces = self.forces(new_params)
        self.expectations = self.compute_expectations(self.effective_forces)
        self.B_matrix = self.fillB(self.expectations)
        self.C_matrix = self.fillC(self.expectations)


    def equilibrium_avN(self, ntraj=100000, dt=0.01, chunk=0.5,
                        tol=1e-3, max_chunks=20):
        """
        Estimate the equilibrium mean abundance before perturbation by direct simulation.
        Runs runStochasticSimulations in chunks, until the ensemble mean stops changing between chunks.
        """
        rng = np.random.default_rng()
        n0 = rng.uniform(1.0, self.old_params["Kc"], size=ntraj) # sample random initial populations
        n = np.asarray(n0, float)
        prev = n.mean()

        for i in range(max_chunks):
            n = self.runStochasticSimulations(dt, chunk, ntraj=ntraj, n0=n,
                                              params=self.old_params, rng=rng, record=False)
            cur = n.mean()
            if abs(cur - prev) <= tol * max(1.0, abs(cur)):
                return cur
            prev = cur
        return cur


    def forces(self, p, avN=None):
        """
        Converts parameters into forces
        """
        r, T, Kc, m = p["r"], p["T"], p["Kc"], p["m"]
        phi1, phi2, S = p["phi1"], p["phi2"], p["S"]

        if avN is None:
            n = self.avN
        else:
            n = avN

        return [
            -r + phi1 * S * n,
            r / T + r / Kc - phi1 - phi2 * S * n,
            phi2 - r / (T * Kc),
            m
        ]

    def compute_expectations(self, effective_forces,
                             xlo=-40.0, xhi=40.0, ngrid=20001):
        """
        Moments of the MaxEnt density for effective forces [a1, a2, a3, a4].

        Density: p(n) ~ (1/n) exp[(2/sigma^2)(a1 n + a2 n^2/2 + a3 n^3/3 + a4 log n)].
        
        Substituting n = exp(x) and shifting with a max-shift (log-sum-exp).
         The max-shift is computer per integral.
         Integrates on a fixed, pre-sampled grid of 20001 points.
        """
        a1, a2, a3, a4 = effective_forces
        k = 2.0 / self.sigma ** 2

        x = np.linspace(xlo, xhi, ngrid)
        ex = np.exp(x)
        G = k * (a1 * ex + a2 * ex ** 2 / 2 + a3 * ex ** 3 / 3 + a4 * x)
        Gmax = G.max()
        logZ = np.log(np.trapezoid(np.exp(G - Gmax), x)) + Gmax

        def E_positive(log_test):  # max-shift for positive functions
            f = G + log_test
            M = f.max()
            return np.exp(np.log(np.trapezoid(np.exp(f - M), x)) + M - logZ)

        def E_signed(factor):  # max-shift for possibly negative functions
            num = np.trapezoid(factor * np.exp(G - Gmax), x)
            den = np.trapezoid(np.exp(G - Gmax), x)
            return num / den

        exp = {f"n{j}": E_positive(j * x) for j in range(1, 7)}
        exp["1/n"] = E_positive(-x)
        exp["logn"] = E_signed(x)
        exp["log2n"] = E_signed(x ** 2)
        exp["nlogn"] = E_signed(ex * x)
        exp["n2logn"] = E_signed(ex ** 2 * x)
        exp["n3logn"] = E_signed(ex ** 3 * x)

        return exp

    # ---- Draw initial abundances from the MaxEnt density --------------
    def _sample_from_maxent(self, forces, n_samples, rng,
                            xlo=-40.0, xhi=40.0, ngrid=20001):
        """
        Inverse-CDF sampling of n ~ MaxEnt(forces), reusing the same
        x = log n grid the moment quadrature uses.
        """
        a1, a2, a3, a4 = forces
        k = 2.0 / self.sigma ** 2
        x = np.linspace(xlo, xhi, ngrid)
        ex = np.exp(x)
        G = k * (a1 * ex + a2 * ex ** 2 / 2 + a3 * ex ** 3 / 3 + a4 * x)
        w = np.exp(G - G.max())
        cdf = np.cumsum(w)
        cdf /= cdf[-1]
        x_samp = np.interp(rng.random(n_samples), cdf, x)
        return np.exp(x_samp)

    # ---- Empirical moments of an ensemble -----------------------------
    @staticmethod
    def _empirical_moments(n):
        e = {f"n{j}": np.mean(n ** j) for j in range(1, 4)}
        nl = n[n > 0]
        ln = np.log(nl)
        e["logn"] = np.mean(ln)
        return e


    #####################################################################
    ###                              DME                              ###
    #####################################################################
    def fillB(self, expectations):
        """B matrix of the DME update."""
        e = expectations
        n1, n2, n3, n4, n5 = e["n1"], e["n2"], e["n3"], e["n4"], e["n5"]
        return np.array([
            [n1, n2, n3, 1.0],
            [n2, n3, n4, n1],
            [n3, n4, n5, n2],
            [1.0, n1, n2, e["1/n"]],
        ])

    def fillC(self, expectations):
        """C = (2/sigma^2) * Cov(A), with A = (n, n^2/2, n^3/3, log n)"""
        e = expectations
        cov = np.array([
            [e["n2"] - e["n1"] ** 2,
             0.5 * (e["n3"] - e["n1"] * e["n2"]),
             (1 / 3) * (e["n4"] - e["n1"] * e["n3"]),
             e["nlogn"] - e["n1"] * e["logn"]],
            [0.5 * (e["n3"] - e["n2"] * e["n1"]),
             0.25 * (e["n4"] - e["n2"] ** 2),
             (1 / 6) * (e["n5"] - e["n2"] * e["n3"]),
             0.5 * (e["n2logn"] - e["n2"] * e["logn"])],
            [(1 / 3) * (e["n4"] - e["n3"] * e["n1"]),
             (1 / 6) * (e["n5"] - e["n3"] * e["n2"]),
             (1 / 9) * (e["n6"] - e["n3"] ** 2),
             (1 / 3) * (e["n3logn"] - e["n3"] * e["logn"])],
            [e["nlogn"] - e["logn"] * e["n1"],
             0.5 * (e["n2logn"] - e["logn"] * e["n2"]),
             (1 / 3) * (e["n3logn"] - e["logn"] * e["n3"]),
             e["log2n"] - e["logn"] ** 2],
        ])
        return (2.0 / self.sigma ** 2) * cov

    @staticmethod
    def _solve(C, rhs, update="regularised", rcond=1e-4):
        """Solve the DME linear system  C @ delta = rhs  for the force update.

        update="plain"       : np.linalg.solve, straightforward inverse.

        update="regularised" : Jacobi-scaled least squares with singular-value
                               cutoff rcond (default). Stable, but the a4
                               endpoint is set by rcond, not by the data.

        update="fixed_a4"    : hold a4 constant; solve only the 3x3 (a1,a2,a3)
                               block (with update = "regularised")
        """
        if update == "plain":
            return np.linalg.solve(C, rhs)

        if update == "regularised":
            d = np.sqrt(np.abs(np.diag(C)));
            d[d == 0] = 1.0
            Dinv = np.diag(1.0 / d)
            xt, *_ = np.linalg.lstsq(Dinv @ C @ Dinv, Dinv @ rhs, rcond=rcond)
            return Dinv @ xt

        if update == "fixed_a4":
            delta = np.zeros(4)
            d = np.sqrt(np.abs(np.diag(C[:3, :3])));
            d[d == 0] = 1.0
            Dinv = np.diag(1.0 / d)
            xt, *_ = np.linalg.lstsq(Dinv @ C[:3, :3] @ Dinv, Dinv @ rhs[:3], rcond=rcond)
            delta[:3] = Dinv @ xt
            return delta

        raise ValueError(f"unknown update rule {update!r}")

    def runDME(self, dt, t, update="regularised",
               rcond=1e-4, rtol=1e-6, atol=1e-9):
        """
        Relax the effective forces toward the target forces via

            d(eff)/dt = C^{-1} B (target - eff).

        update : linear-solve rule inside the RHS -- 'plain', 'regularised' (default), or 'fixed_a4' (see _solve).

        method : integrator -- 'euler' (fixed step dt) or any solve_ivp method
                 ('LSODA' default, 'Radau', 'BDF', ...).  dt is the fixed step
                 for euler and the output spacing otherwise.
        """
        eff0 = np.asarray(self.effective_forces, float)

        def rhs(_t, eff):
            e = self.compute_expectations(eff)
            tf = np.asarray(self.forces(self.new_params, e["n1"]))
            return self._solve(self.fillC(e), self.fillB(e) @ (tf - eff),
                               update=update, rcond=rcond)

        t_eval = np.arange(self.time, self.time + t + 1e-12, dt)
        sol = solve_ivp(rhs, (self.time, self.time + t), eff0, method='LSODA', t_eval=t_eval, rtol=rtol, atol=atol)
        sol_t, sol_y = sol.t, sol.y.T

        for tk, ec in zip(sol_t, sol_y):
            e = self.compute_expectations(ec)
            self.dme_history.append({"time": float(tk),
                                     "effective_forces": list(ec),
                                     "target_forces": self.forces(self.new_params, e["n1"]),
                                     "expectations": e})
        last = self.dme_history[-1]
        self.time = last["time"];
        self.effective_forces = last["effective_forces"]
        self.target_forces = last["target_forces"];
        self.expectations = last["expectations"]
        self.B_matrix = self.fillB(self.expectations);
        self.C_matrix = self.fillC(self.expectations)
        return self.dme_history

    def runStochasticSimulations(self, dt, t, ntraj=100000, n0=None,
                                 params=None, save_every=1, n_save_full=5, rng=None, record=True):
        """
        Euler-Maruyama simulation of the *true* (new_params) SDE

            dn = mu(n; <n>) dt + sigma sqrt(n) dW,
            mu = a1 n + a2 n^2 + a3 n^3 + a4,
            a = forces(params, <n>),

        started from the old equilibrium (n0 ~ MaxEnt(effective_forces)).
        Appends {time, expectations} to self.stochastic_history at every saved step.
        """
        rng = np.random.default_rng() if rng is None else rng
        params = self.new_params if params is None else params

        if n0 is None:
            n0 = self._sample_from_maxent(self.effective_forces, ntraj, rng)

        n = np.asarray(n0, dtype=float).copy()

        nsteps = int(np.floor(t / dt))
        sqdt = np.sqrt(dt)

        try:
            obs_time = self.stochastic_history[-1]['time']
        except:
            obs_time = 0.0

        if record:
            self.stochastic_history.append(
                {"time": obs_time, "expectations": self._empirical_moments(n), "populations": n[:20]})

        for j in range(1, nsteps + 1):
            obs_time += dt

            avN = n.mean()
            a1, a2, a3, a4 = self.forces(params, avN)
            mu = a1 * n + a2 * n ** 2 + a3 * n ** 3 + a4
            dW = rng.normal(0.0, sqdt, size=n.size)
            n = n + mu * dt + self.sigma * np.sqrt(np.maximum(n, 0.0)) * dW
            np.maximum(n, 0.0, out=n)

            if j % save_every == 0 and record:
                self.stochastic_history.append(
                    {"time": obs_time, "expectations": self._empirical_moments(n), "populations": n[:20]})

            if j % np.ceil(nsteps / n_save_full) == 0 and record:
                self.full_stochastic_distributions.append(
                    {"time": obs_time, "populations": n}
                )

        self.avN = n.mean()
        return n

    def runDyMES(self, dt, tmax):
        return None

    def to_dataframes(self):
        """Return (dme_df, stoch_df).  Each row is one saved time step;
        expectation keys become columns.  dme_df also carries the effective
        and target forces as eff_a1..eff_a4 / target_a1..target_a4."""

        def expand(history, with_forces=False):
            rows = []
            for rec in history:
                row = {"time": rec["time"], **rec["expectations"]}
                if with_forces:
                    for i in range(4):
                        row[f"eff_a{i + 1}"] = rec["effective_forces"][i]
                        row[f"target_a{i + 1}"] = rec["target_forces"][i]
                rows.append(row)
            return pd.DataFrame(rows)

        return expand(self.dme_history, True), expand(self.stochastic_history, False)

    def plot_moments(self, keys=("n1", "n2", "n3", "logn"), ncols=2,
                     n_traj=20, figsize=None):
        """DME moments (dark line) over the stochastic moments (green line), with a
        few individual stochastic trajectories shown see-through.  Each panel shows
        the per-trajectory version of that panel's statistic (n^k, log n, ...)."""
        dme_df, stoch_df = self.to_dataframes()
        pretty = {"n1": r"$\langle n \rangle$", "n2": r"$\langle n^2 \rangle$",
                  "n3": r"$\langle n^3 \rangle$", "n4": r"$\langle n^4 \rangle$",
                  "logn": r"$\langle \log n \rangle$",
                  "log2n": r"$\langle \log^2 n \rangle$", "1/n": r"$\langle 1/n \rangle$"}

        # individual trajectories saved in stochastic_history as populations = n[:k]
        have_traj = bool(self.stochastic_history) and "populations" in self.stochastic_history[0]
        if have_traj:
            t_traj = np.array([rec["time"] for rec in self.stochastic_history])
            pops = np.array([rec["populations"] for rec in self.stochastic_history])[:, :n_traj]

        def per_traj(key, n):
            """per-trajectory value of the statistic whose mean is `key`."""
            if key[0] == "n" and key[1:].isdigit():
                return n ** int(key[1:])
            npos = np.where(n > 0, n, np.nan)  # guard log / 1/n at n=0
            return {"1/n": 1.0 / npos, "logn": np.log(npos), "log2n": np.log(npos) ** 2,
                    "nlogn": n * np.log(npos), "n2logn": n ** 2 * np.log(npos),
                    "n3logn": n ** 3 * np.log(npos)}.get(key)

        keys = list(keys)
        nrows = int(np.ceil(len(keys) / ncols))
        fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                                 figsize=figsize or (5 * ncols, 3.2 * nrows))
        for ax, key in zip(axes.flat, keys):
            if have_traj:
                Y = per_traj(key, pops)  # shape (n_times, n_traj)
                if Y is not None:
                    ax.plot(t_traj, Y, "-", color="#66c2a5", alpha=0.2, lw=0.8)  # each column = one trajectory
            if key in stoch_df:
                ax.plot(stoch_df["time"], stoch_df[key], "-", lw=2, color="#66c2a5", label="stochastic")
            ax.plot(dme_df["time"], dme_df[key], "-", lw=2, color="#4c4c4c", label="DME")
            ax.set(xlabel="time", ylabel=pretty.get(key, key), title=pretty.get(key, key))
        for ax in axes.flat[len(keys):]:
            ax.axis("off")
        axes.flat[0].legend(frameon=False)
        fig.tight_layout()
        return fig

    def plot_forces(self, figsize=None):
        """DME effective (solid) vs target (dashed) forces over time, one panel
        per force in a 2x2 grid.  Each force gets one colour."""
        dme_df, _ = self.to_dataframes()
        colors = plt.get_cmap("Set2").colors[:4]
        labels = [r"$a_1$ (n)", r"$a_2$ ($n^2/2$)",
                  r"$a_3$ ($n^3/3$)", r"$a_4$ ($\log n$)"]
        fig, axes = plt.subplots(2, 2, figsize=figsize or (10, 7))
        for i, ax in enumerate(axes.flat):
            ax.plot(dme_df["time"], dme_df[f"eff_a{i + 1}"], "-",
                    color=colors[i], lw=2, label="effective")
            ax.plot(dme_df["time"], dme_df[f"target_a{i + 1}"], "--",
                    color=colors[i], lw=2, label="target")
            ax.set(title=labels[i], xlabel="time", ylabel="force")
            ax.legend(frameon=False)
        fig.tight_layout()
        return fig

    def _maxent_pdf(self, n, forces, xlo=-40.0, xhi=40.0, ngrid=20001):
        """Normalized MaxEnt density p(n) for a force vector, using the same
        n = e^x normalizer as compute_expectations."""
        a1, a2, a3, a4 = forces
        k = 2.0 / self.sigma ** 2
        x = np.linspace(xlo, xhi, ngrid);
        ex = np.exp(x)
        G = k * (a1 * ex + a2 * ex ** 2 / 2 + a3 * ex ** 3 / 3 + a4 * x)
        logZ = np.log(np.trapezoid(np.exp(G - G.max()), x)) + G.max()
        n = np.asarray(n, float)
        logp = -np.log(n) + k * (a1 * n + a2 * n ** 2 / 2 + a3 * n ** 3 / 3 + a4 * np.log(n)) - logZ
        return np.exp(logp)

    def plot_distributions(self, bins=40, npdf=500, figsize=None):
        """Stochastic histograms (bars) overlaid with the DME MaxEnt p(n) (lines).
        Each saved time gets one colour; the p(n) from that time's effective forces
        is drawn on top in the same colour."""
        dists = self.full_stochastic_distributions

        # x-range and shared bins come from the stochastic populations
        allpops = np.concatenate([np.asarray(d["populations"], float) for d in dists])
        lo, hi = allpops.min(), allpops.max()
        edges = np.linspace(lo, hi, bins + 1)
        n_grid = np.linspace(max(lo, 1e-9), hi, npdf)

        times = [d["time"] for d in dists]
        tmin, tmax = min(times), max(times)
        cmap = plt.get_cmap("viridis")
        color_for = lambda t: cmap((t - tmin) / (tmax - tmin) if tmax > tmin else 0.5)

        fig, ax = plt.subplots(figsize=figsize or (9, 5.5))
        for d in dists:
            t = d["time"];
            c = color_for(t)
            ax.hist(np.asarray(d["populations"], float), bins=edges,
                    density=True, color=c, alpha=0.3)
            rec = min(self.dme_history, key=lambda r: abs(r["time"] - t))  # DME forces at this time
            ax.plot(n_grid, self._maxent_pdf(n_grid, rec["effective_forces"]),
                    "-", color=c, lw=2.5, label=f"t = {t:.3g}")
        ax.set(xlabel="n", ylabel="probability density",
               title="Stochastic distributions (bars) vs DME MaxEnt p(n) (lines)")
        ax.set_xlim(lo, hi)
        ax.legend(frameon=False, title="time")
        fig.tight_layout()
        return fig
