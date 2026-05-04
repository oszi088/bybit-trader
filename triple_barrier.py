"""
Triple Barrier labeling (Lopez de Prado, AFML 3. fejezet).

A hagyományos "következő gyertya iránya" label helyett dinamikus barrier-ek:
  +1  ha a felső barrier (TP) érintődik meg először
  -1  ha az alsó barrier (SL) érintődik meg először
   0  ha a függőleges barrier (időlimit) érintődik meg először (döntetlen)

Az ATR-alapú barrier-ek kompatibilisek a meglévő StopConfig-gal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import StopConfig


def make_labels(
    ohlcv: pd.DataFrame,
    atr: pd.Series,
    stops: StopConfig,
    max_holding: int = 20,
) -> pd.DataFrame:
    """
    Triple barrier label minden gyertyához.

    Paraméterek:
        ohlcv        -- OHLCV DataFrame idő-indexszel
        atr          -- ATR Series (ugyanolyan index)
        stops        -- StopConfig (atr_tp_mult, atr_stop_mult)
        max_holding  -- max gyertyák száma amíg várunk (függőleges barrier)

    Visszatér:
        DataFrame: label (-1/0/+1), holding_period, ret (becsült hozam)
    """
    closes = ohlcv["close"].values
    highs  = ohlcv["high"].values
    lows   = ohlcv["low"].values
    atrs   = atr.values
    n = len(closes)

    labels  = np.zeros(n, dtype=np.int8)
    holding = np.zeros(n, dtype=np.int16)
    rets    = np.zeros(n, dtype=np.float32)

    for i in range(n - 1):
        entry   = closes[i]
        atr_val = atrs[i]
        if np.isnan(atr_val) or atr_val <= 0 or entry <= 0:
            continue

        tp = entry + stops.atr_tp_mult   * atr_val
        sl = entry - stops.atr_stop_mult * atr_val

        result   = 0
        hold_len = 0
        exit_ret = 0.0

        for j in range(i + 1, min(i + 1 + max_holding, n)):
            hold_len = j - i
            tp_hit = highs[j] >= tp
            sl_hit = lows[j] <= sl
            if tp_hit and sl_hit:
                # Mindkét barrier ugyanazon a gyertyán — konzervatív: SL nyer.
                # A gyertya belső sorrendja ismeretlen; worst-case feltételezés
                # csökkenti az optimista label-torzítást.
                result   = -1
                exit_ret = (sl - entry) / entry
                break
            if tp_hit:
                result   = 1
                exit_ret = (tp - entry) / entry
                break
            if sl_hit:
                result   = -1
                exit_ret = (sl - entry) / entry
                break
        else:
            # Függőleges barrier: záróár az időlimit végén
            end_idx  = min(i + max_holding, n - 1)
            hold_len = end_idx - i
            exit_ret = (closes[end_idx] - entry) / entry

        labels[i]  = result
        holding[i] = hold_len
        rets[i]    = exit_ret

    return pd.DataFrame(
        {"label": labels, "holding_period": holding, "ret": rets},
        index=ohlcv.index,
    )
