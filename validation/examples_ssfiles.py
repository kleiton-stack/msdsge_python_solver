"""Steady-state files of the RISE examples, ported to Python by hand.

Interface: ``f(model, p) -> dict``, with ``p`` a dictionary {parameter: value} for the regime. The
returned dictionary holds the endogenous variables and, if any, computed parameters (RISE's
``newp``). Optional function attributes: ``f.imposed = True`` (do not check) and ``f.unique = True``.
Usage: ``model.solve(steady_state_file=f)``.
"""
import numpy as np


def fs2000(model, p):
    """Port of ``examples/NonlinearModels/fs2000_steadystate.m`` (Schorfheide, 2000)."""
    alp, bet, gam, mst, psi, dlt = p["alp"], p["bet"], p["gam"], p["mst"], p["psi"], p["del"]
    dA = np.exp(gam)
    gst = 1 / dA
    m = mst
    khst = ((1 - gst * bet * (1 - dlt)) / (alp * gst ** alp * bet)) ** (1 / (alp - 1))
    xist = (((khst * gst) ** alp - (1 - gst * (1 - dlt)) * khst) / mst) ** (-1)
    nust = psi * mst ** 2 / ((1 - alp) * (1 - psi) * bet * gst ** alp * khst ** alp)
    n = xist / (nust + xist)
    P = xist + nust
    k = khst * n
    l = psi * mst * n / ((1 - psi) * (1 - n))
    c = mst / P
    d = l - mst + 1
    y = k ** alp * n ** (1 - alp) * gst ** alp
    R = mst / bet
    W = l / n
    return dict(m=m, P=P, c=c, e=1.0, W=W, R=R, k=k, d=d, n=n, l=l, gy_obs=np.log(dA), gp_obs=np.log(m / dA), y=y, dA=dA)


def canonical_const(model, p):
    """Port of ``examples/ModelsWithSteadyStateFile/steady_state_4_Canonical_Const.m``."""
    return dict(DPQ_P_NW=p["paiss"], Y=0.0, ZGDP=p["gyss"], ZI=0.0, ZPAI=0.0, ZY=0.0, D_GDP_NW=p["gyss"],
                I=0.0, PAI=0.0, R=0.0, RN3M_NW=p["iss"])


FILES = {"fs2000": fs2000, "canonical_const": canonical_const}
