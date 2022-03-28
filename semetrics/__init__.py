import os
import logging
import oct2py
import numpy as np
from pesq import pesq

logging.basicConfig(level=logging.ERROR)
oc = oct2py.Oct2Py(logger=logging.getLogger())

COMPOSITE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "composite.m")


def pesq_mos(reference: np.ndarray, degraded: np.ndarray, sr:int):
    mode = "nb" if sr < 16000 else "wb"
    return pesq(sr, reference, degraded, mode)


def composite(reference: np.ndarray, degraded: np.ndarray, sr:int, mp: bool = False):
    assert reference.ndim == 1
    assert degraded.ndim == 1
    assert sr in (8000, 16000)
    pesq_score = pesq_mos(reference, degraded, sr)
    reference = reference.reshape(-1, 1)
    degraded = degraded.reshape(-1, 1)
    if mp:
        oc_local = oct2py.Oct2Py(logger=logging.getLogger())
    else:
        oc_local = oc
    try:
        csig, cbak, covl, ssnr = oc_local.feval(COMPOSITE, reference, degraded, sr, nout=4)
    except Exception as e:
        print("Error during composite computation:", e)
        raise e
    if mp:
        oc_local.exit()
    csig += 0.603 * pesq_score
    cbak += 0.478 * pesq_score
    covl += 0.805 * pesq_score
    return pesq_score, csig, cbak, covl, ssnr
