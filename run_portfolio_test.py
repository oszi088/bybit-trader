"""
Portfolio + scale-in tesztek — 1 nap (86 400 bar), előszámított cache + pointer opt.

Diagnózis:
  * 2min ATR ~$1, round-trip fee = $193/BTC -> ATR stop ertelmetlen 1s-en.
  * Fix % stop (SL=0.3%, TP=0.9%): fee=0.2% < SL=0.3% -> életképes.
  * thr=0.30 -> 13% bar = 1 kereskedés 8s-ként -> fee halál.
  * thr=0.50 -> 2.5% bar, thr=0.55 -> 1.1% bar -> kevesebb, jobb jelek.

Gyorsítás:
  * entry-indexek előreszámítva; bisect -> O(log N) keresés
  * next_dca_bar cache a pozíción -> O(1) DCA lookup (nem O(N-i) rescan!)
  * vectorized equity fill a skip-ahead-ben
"""
import bisect
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from config import (
    TradingConfig, ScaleInConfig,
    HFT_INDICATORS, HFT_WEIGHTS,
    DEFAULT_WEIGHTS, TREND_WEIGHTS, RANGE_WEIGHTS,
)
from agent import TradingAgent
from signals import compute_signal_matrix, compute_scores_with_regime
from backtest import (
    _apply_slippage, _close_position, _find_exit_bar_long,
    Trade, _OpenPos, _max_drawdown,
)

# ---------------------------------------------------------------------------
SYMBOLS  = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
DAY_BARS = 86_400

# ---------------------------------------------------------------------------
print("Adatok betoltese...", flush=True)
raw: Dict[str, pd.DataFrame] = {}
for sym in SYMBOLS:
    df = pd.read_csv(f"data/{sym}_1s_202412_202412_binance.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    raw[sym] = df.set_index("timestamp").head(DAY_BARS)
    print(f"  {sym}: {DAY_BARS:,} bar", flush=True)

# ---------------------------------------------------------------------------
base_cfg = TradingConfig(
    initial_balance=10_000, fee_rate=0.001, timeframe="1s", buy_threshold=0.50)
base_cfg.indicators = HFT_INDICATORS
base_cfg.weights    = dict(HFT_WEIGHTS)

print("\nPre-compute (indikatorok + score-ok)...", flush=True)
t0 = time.perf_counter()
precomputed: Dict[str, dict] = {}
for sym in SYMBOLS:
    agent    = TradingAgent(base_cfg)
    enriched = agent.prepare(raw[sym])
    fg       = agent._fg_for_ts(enriched.index[0])
    sig      = compute_signal_matrix(enriched, base_cfg.indicators, fg)
    scores   = compute_scores_with_regime(
        sig, enriched, base_cfg.regime,
        DEFAULT_WEIGHTS, TREND_WEIGHTS, RANGE_WEIGHTS)
    precomputed[sym] = {
        "enriched": enriched,
        "scores":   scores,
        "lows":     enriched["low"].values.astype(np.float64),
        "highs":    enriched["high"].values.astype(np.float64),
        "closes":   enriched["close"].values.astype(np.float64),
        "atrs":     (enriched["atr"].values.astype(np.float64)
                     if "atr" in enriched.columns else np.zeros(len(enriched))),
        "n":        len(enriched),
    }
    print(f"  {sym}: score [{scores.min():.2f}, {scores.max():.2f}]", flush=True)
print(f"  Osszesen: {time.perf_counter()-t0:.1f}s\n", flush=True)


# ---------------------------------------------------------------------------
def run_config(name, syms, max_c, fee, thr, sl_pct, tp_pct,
               pos_size=0.20, sc: Optional[ScaleInConfig] = None):
    cfg = TradingConfig(
        initial_balance=10_000, fee_rate=fee,
        position_size=pos_size,
        buy_threshold=thr, sell_threshold=-thr,
        timeframe="1s",
    )
    cfg.stops.use_atr_stops     = False
    cfg.stops.stop_loss_pct     = sl_pct
    cfg.stops.take_profit_pct   = tp_pct
    cfg.stops.use_trailing_stop = False
    cfg.risk.use_fixed_risk_sizing   = False
    cfg.risk.score_proportional_size = False
    sc = sc or ScaleInConfig()

    sym_data = {s: precomputed[s] for s in syms}
    bt_cfg   = cfg.backtest
    stops    = cfg.stops
    n        = min(d["n"] for d in sym_data.values())
    common_index = sym_data[syms[0]]["enriched"].index[:n]

    # ── Pointer-alapú entry-index előszámítás ────────────────────────────
    entry_bars: Dict[str, np.ndarray] = {
        s: np.where(sym_data[s]["scores"] >= thr)[0]
        for s in syms
    }

    def _next_entry_after(sym: str, bar: int) -> Optional[int]:
        """Legközelebbi entry bar > bar, O(log N)."""
        arr = entry_bars[sym]
        idx = bisect.bisect_right(arr, bar)
        return int(arr[idx]) if idx < len(arr) else None

    # ── DCA trigger cache  (elkerüli az O(N-i) rescan-t minden iterációban) ──
    def _compute_next_dca(sym: str, buy_price: float, start: int) -> Optional[int]:
        """Következő scale-in trigger bar kiszámítása — egyszer, cache-elve."""
        if not sc.enabled:
            return None
        closes = sym_data[sym]["closes"]
        seg = closes[start:n]
        if sc.mode == "dca":
            h = np.where(seg <= buy_price * (1.0 - sc.trigger_pct))[0]
        else:
            h = np.where(seg >= buy_price * (1.0 + sc.trigger_pct))[0]
        return int(start + h[0]) if len(h) else None

    # ── Állapot ──────────────────────────────────────────────────────────
    cash      = float(cfg.initial_balance)
    open_pos: Dict[str, _OpenPos] = {}
    all_trades: List[Trade] = []
    per_symbol = {s: [] for s in syms}
    equity_arr = np.full(n, np.nan, dtype=np.float64)
    i = 0

    def _mtm(bar: int) -> float:
        return sum(
            p.size * float(sym_data[p.sym]["closes"][bar])
            for p in open_pos.values())

    def _open(sym: str, bar: int, score: float) -> None:
        nonlocal cash
        d     = sym_data[sym]
        price = float(d["closes"][bar])
        fill  = _apply_slippage(price, "BUY", bt_cfg)
        sl    = fill * (1.0 - stops.stop_loss_pct)
        tp    = fill * (1.0 + stops.take_profit_pct)
        frac  = sc.first_tranche_pct if sc.enabled else pos_size
        alloc = cash * frac
        size  = alloc / (fill * (1.0 + cfg.fee_rate))
        cost  = size * fill * (1.0 + cfg.fee_rate)
        if size <= 0 or cost > cash:
            return
        cash -= cost
        trade = Trade(
            entry_time=d["enriched"].index[bar],
            entry_price=fill, size=size, direction="long")
        ndb = _compute_next_dca(sym, fill, bar + 1)   # <- cache első DCA bar-t
        open_pos[sym] = _OpenPos(
            sym=sym, direction="long", entry_i=bar,
            sl=sl, tp=tp, size=size, trade=trade,
            tranche_idx=1, last_buy_price=fill,
            next_dca_bar=ndb)

    def _close(sym: str, pos: _OpenPos, fill: float,
               bar: int, reason: str) -> None:
        nonlocal cash
        d  = sym_data[sym]
        ts = d["enriched"].index[bar]
        cash = _close_position(pos.trade, fill, ts, cash, cfg, bt_cfg, reason)
        all_trades.append(pos.trade)
        per_symbol[sym].append(pos.trade)
        del open_pos[sym]

    t_start = time.perf_counter()

    while i < n:
        # ── Kilépések ────────────────────────────────────────────────────
        to_cl = []
        for sym, pos in open_pos.items():
            d = sym_data[sym]
            if float(d["lows"][i]) <= pos.sl:
                to_cl.append((sym, pos, pos.sl, "stop_loss"))
            elif float(d["highs"][i]) >= pos.tp:
                to_cl.append((sym, pos, pos.tp, "take_profit"))
        for sym, pos, fill, reason in to_cl:
            _close(sym, pos, fill, i, reason)

        # ── Scale-in ─────────────────────────────────────────────────────
        if sc.enabled:
            for sym, pos in list(open_pos.items()):
                if pos.tranche_idx >= sc.n_tranches:
                    continue
                curr = float(sym_data[sym]["closes"][i])
                if sc.mode == "dca":
                    triggered = curr <= pos.last_buy_price * (1.0 - sc.trigger_pct)
                else:
                    triggered = curr >= pos.last_buy_price * (1.0 + sc.trigger_pct)
                if triggered:
                    alloc = cash * sc.add_tranche_pct
                    af    = _apply_slippage(curr, "BUY", bt_cfg)
                    asiz  = alloc / (af * (1.0 + cfg.fee_rate))
                    cost  = asiz * af * (1.0 + cfg.fee_rate)
                    if asiz > 0 and cost <= cash:
                        cash -= cost
                        new_total = pos.size + asiz
                        # Súlyozott átlag entry ár — _close_position így helyes PnL-t számol
                        pos.trade.entry_price = (
                            pos.trade.size * pos.trade.entry_price + asiz * af
                        ) / new_total
                        pos.trade.size = new_total   # trade.size szinkronban pos.size-zal
                        pos.size          = new_total
                        pos.tranche_idx   += 1
                        pos.last_buy_price = af
                        # Cache újraszámítás — csak ha még van tranche
                        if pos.tranche_idx < sc.n_tranches:
                            pos.next_dca_bar = _compute_next_dca(sym, af, i + 1)
                        else:
                            pos.next_dca_bar = None

        # ── Új belépések ─────────────────────────────────────────────────
        while len(open_pos) < max_c:
            avail = [s for s in syms if s not in open_pos]
            if not avail:
                break
            best  = max(avail, key=lambda s: float(sym_data[s]["scores"][i]))
            bscr  = float(sym_data[best]["scores"][i])
            if bscr < thr:
                break
            _open(best, i, bscr)
            if best not in open_pos:
                break

        equity_arr[i] = cash + _mtm(i)

        # ── Skip-ahead (O(log N) / event, DCA O(1) via cache) ─────────────
        if not open_pos:
            mn = n
            for sym in syms:
                ne = _next_entry_after(sym, i)
                if ne is not None and ne < mn:
                    mn = ne
            if mn >= n:
                equity_arr[i:] = cash
                break
            equity_arr[i + 1:mn] = cash
            i = mn

        else:
            events = []

            for sym, pos in open_pos.items():
                d = sym_data[sym]
                ex_i, ex_r, ex_f = _find_exit_bar_long(
                    d["lows"], d["highs"], pos.sl, pos.tp, i + 1, n)
                if ex_i is None:
                    ex_i = n - 1
                    ex_r = "max_holding"
                    ex_f = float(d["closes"][n - 1])
                events.append((ex_i, "exit", sym, ex_f))

                # DCA: cached bar, O(1) lookup (nem O(N-i) rescan!)
                if (sc.enabled
                        and pos.tranche_idx < sc.n_tranches
                        and pos.next_dca_bar is not None
                        and pos.next_dca_bar > i):
                    events.append((pos.next_dca_bar, "scale", sym,
                                   float(d["closes"][pos.next_dca_bar])))

            if len(open_pos) < max_c:
                for sym in syms:
                    if sym not in open_pos:
                        ne = _next_entry_after(sym, i)
                        if ne is not None:
                            events.append((ne, "entry", sym,
                                           float(sym_data[sym]["scores"][ne])))

            if not events:
                equity_arr[i:] = cash + _mtm(i)
                break

            nxt = min(ev[0] for ev in events)
            # Ha nincs esemény az aktuális bar UTÁN (utolsó bar edge case), kilépés
            if nxt <= i:
                equity_arr[i:] = cash + _mtm(i)
                break
            if nxt > i + 1:
                eq_seg = np.full(nxt - i - 1, cash)
                for sym, pos in open_pos.items():
                    eq_seg += pos.size * sym_data[sym]["closes"][i + 1:nxt]
                equity_arr[i + 1:nxt] = eq_seg
            i = nxt

    # Maradék kitöltés
    if np.any(np.isnan(equity_arr)):
        last = cash + _mtm(min(i, n - 1)) if i < n else cash
        equity_arr[np.isnan(equity_arr)] = last

    for sym, pos in list(open_pos.items()):
        _close(sym, pos, float(sym_data[sym]["closes"][n - 1]),
               n - 1, "end_of_data")
    equity_arr[-1] = cash

    elapsed = time.perf_counter() - t_start
    eq  = pd.Series(equity_arr, index=common_index)
    ret = (cash / 10_000 - 1) * 100
    dd  = _max_drawdown(eq)
    ntr = len(all_trades)
    wr  = sum(1 for t in all_trades if t.pnl > 0) / ntr * 100 if ntr else 0
    avg_pnl = sum(t.pnl for t in all_trades) / ntr if ntr else 0
    durs = [(t.exit_time - t.entry_time).total_seconds()
            for t in all_trades if t.exit_time]
    avg_dur = sum(durs) / len(durs) if durs else 0

    print(f"\n{'='*65}", flush=True)
    print(f" {name}", flush=True)
    print(f" Fee={fee*100:.2f}%  SL={sl_pct*100:.2f}%  TP={tp_pct*100:.2f}%  "
          f"kuszob={thr}  max_c={max_c}", flush=True)
    print(f" Trades:{ntr:>5,}  WinRate:{wr:>5.1f}%  Return:{ret:>+7.2f}%  "
          f"MaxDD:{dd:>6.1f}%  Fut:{elapsed:.1f}s", flush=True)
    print(f" Avg PnL/trade: ${avg_pnl:>+.3f}  Avg tartam: {avg_dur:.0f}s", flush=True)
    if len(syms) > 1:
        for sym in syms:
            ts = per_symbol.get(sym, [])
            if not ts:
                continue
            sw = sum(1 for t in ts if t.pnl > 0) / len(ts) * 100
            sp = sum(t.pnl for t in ts)
            print(f"   {sym:<12} {len(ts):>4} trade  "
                  f"WR{sw:>5.1f}%  PnL${sp:>+8.2f}", flush=True)


# ---------------------------------------------------------------------------
run_config(
    "1. BTC egyedul | fix SL=0.3% TP=0.9% | thr=0.30 | fee=0.1%  [tul sok jel]",
    ["BTCUSDT"], 1, fee=0.001, thr=0.30, sl_pct=0.003, tp_pct=0.009)

run_config(
    "2. BTC egyedul | fix SL=0.3% TP=0.9% | thr=0.50 | fee=0.1%",
    ["BTCUSDT"], 1, fee=0.001, thr=0.50, sl_pct=0.003, tp_pct=0.009)

run_config(
    "3. BTC egyedul | fix SL=0.2% TP=0.6% | thr=0.50 | fee=0.02% [maker]",
    ["BTCUSDT"], 1, fee=0.0002, thr=0.50, sl_pct=0.002, tp_pct=0.006)

run_config(
    "4. Top5 parhuzamos max=2 | SL=0.3% TP=0.9% | thr=0.55 | fee=0.1%",
    SYMBOLS, 2, fee=0.001, thr=0.55, sl_pct=0.003, tp_pct=0.009)

sc_dca = ScaleInConfig(
    enabled=True, mode="dca", n_tranches=2,
    trigger_pct=0.003, first_tranche_pct=0.40, add_tranche_pct=0.35)
run_config(
    "5. Top5 max=3 + DCA 2x | SL=0.4% TP=1.2% | thr=0.55 | fee=0.1%",
    SYMBOLS, 3, fee=0.001, thr=0.55, sl_pct=0.004, tp_pct=0.012, sc=sc_dca)

sc_pyr = ScaleInConfig(
    enabled=True, mode="pyramid", n_tranches=2,
    trigger_pct=0.003, first_tranche_pct=0.30, add_tranche_pct=0.25)
run_config(
    "6. Top5 max=3 + Pyramid | SL=0.2% TP=0.6% | thr=0.55 | fee=0.02% [maker]",
    SYMBOLS, 3, fee=0.0002, thr=0.55, sl_pct=0.002, tp_pct=0.006, sc=sc_pyr)

print("\nKesz.")
