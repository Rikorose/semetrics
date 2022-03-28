import os
import logging
import oct2py
import numpy as np
from pesq import pesq

logging.basicConfig(level=logging.ERROR)
oc = oct2py.Oct2Py(logger=logging.getLogger())

COMPOSITE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "composite.m")


def pesq_mos(reference: np.ndarray, degraded: np.ndarray, sr: int):
    """Computes PESQ [1] score using the `pesq` python package.

    Args:
        reference (np.ndarray): 1-D numpy array containing a reference audio signal.
        degraded (np.ndarray): 1-D numpy array containing a degraded audio signal.
        sr (int): Sampling rate of both input signal.

    Returns:
        float: PESQ score


    References:
        [1] Rix, A.W., Beerends, J.G., Hollier, M.P. and Hekstra, A.P., 2001, May. Perceptual
            evaluation of speech quality (PESQ)-a new method for speech quality assessment of
            telephone networks and codecs. ICASSP 2001.
    """
    assert reference.ndim == 1
    assert degraded.ndim == 1
    assert sr in (8000, 16000)
    mode = "nb" if sr < 16000 else "wb"
    return pesq(sr, reference, degraded, mode)


def composite(reference: np.ndarray, degraded: np.ndarray, sr: int, mp: bool = False):
    """Comptes composite metrics [1] using octave cli.

    Args:
        reference (np.ndarray): 1-D numpy array containing a reference audio signal.
        degraded (np.ndarray): 1-D numpy array containing a degraded audio signal.
        sr (int): Sampling rate of both input signal.
        mp (bool): Support multi-processing by instanciating new a octave-cli for each call.

    Returns:
        float: PESQ score [2]
        float: CSIG (Composite Signal MOS)
        float: CBAK (Composite Background MOS)
        float: COVL (Composite Overall MOS)
        float: SSNR (Semental signal to noise ratio)

    References:
        [1] Hu, Y. and Loizou, P. (2006). �Evaluation of objective measures for speech enhancement,
            Proceedings of INTERSPEECH-2006, Philadelphia, PA, September 2006.
        [2] Rix, A.W., Beerends, J.G., Hollier, M.P. and Hekstra, A.P., 2001, May. Perceptual
            evaluation of speech quality (PESQ)-a new method for speech quality assessment of
            telephone networks and codecs. ICASSP 2001.
    """
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
