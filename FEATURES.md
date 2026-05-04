# Feature katalógus

Teljes lista a `ml_features_v2.py` által előállított feature-ökről.

A \* jelölt feature-ök opcionálisak — csak akkor kerülnek a mátrixba,
ha a megfelelő adat forrás rendelkezésre áll (funding_df, oi_df, stb.).

---

## 1. Nyers indikátor értékek (22)

| Feature | Forrás | Mit mér |
|---------|--------|---------|
| `rsi` | RSI(14) | Relatív erő index — 0..100 |
| `macd` | MACD(12,26) | MACD vonal |
| `macd_signal` | MACD signal(9) | MACD jelvonal |
| `macd_hist` | MACD histogram | MACD - signal különbsége |
| `stoch_k` | Stochastic %K(14) | Gyors stochastic vonal |
| `stoch_d` | Stochastic %D(3) | Simított stochastic |
| `cci` | CCI(20) | Commodity Channel Index |
| `atr` | ATR(14) | Átlagos valódi tartomány (USD) |
| `adx` | ADX(14) | Trend erősség (irány nélkül) |
| `plus_di` | +DI(14) | Felfelé irányú mozgás indikátor |
| `minus_di` | -DI(14) | Lefelé irányú mozgás indikátor |
| `obv` | OBV | On-Balance Volume |
| `mfi` | MFI(14) | Money Flow Index |
| `sma_fast` | SMA(20) | Gyors mozgóátlag |
| `sma_slow` | SMA(50) | Lassú mozgóátlag |
| `sma_long` | SMA(200) | Hosszú mozgóátlag |
| `ema_fast` | EMA(12) | Gyors exponenciális MA |
| `ema_slow` | EMA(26) | Lassú exponenciális MA |
| `bb_upper` | BB(20,2) felső | Bollinger felső sáv |
| `bb_mid` | BB(20,2) közép | Bollinger középvonal (SMA20) |
| `bb_lower` | BB(20,2) alsó | Bollinger alsó sáv |
| `vwap` | VWAP | Volumen-súlyozott átlagár |

---

## 2. Normalizált pozíció metrikák (13)

| Feature | Képlet | Mit mér |
|---------|--------|---------|
| `bb_pct_b` | (close - bb_lower) / (bb_upper - bb_lower) | Ár helyzete a BB sávban (0=alsó, 1=felső, >1 kitörés) |
| `bb_width_pct` | (bb_upper - bb_lower) / close | BB sáv szélessége relatívan — volatilitás proxy |
| `sma_fast_dist` | (close - sma_fast) / close | Ár távolsága SMA20-tól |
| `sma_slow_dist` | (close - sma_slow) / close | Ár távolsága SMA50-től |
| `sma_long_dist` | (close - sma_long) / close | Ár távolsága SMA200-tól |
| `ema_fast_dist` | (close - ema_fast) / close | Ár távolsága EMA12-től |
| `ema_slow_dist` | (close - ema_slow) / close | Ár távolsága EMA26-tól |
| `vwap_dist` | (close - vwap) / close | Ár távolsága VWAP-tól (intraday referencia) |
| `atr_pct` | atr / close | Relatív volatilitás szint |
| `stoch_kd_diff` | stoch_k - stoch_d | Stochastic kereszteződés erőssége |
| `macd_hist_delta` | macd_hist.diff(1) | MACD histogram változása (gyorsulás) |
| `macd_hist_accel` | macd_hist_delta.diff(1) | MACD histogram gyorsulásának változása |
| `di_diff_norm` | (plus_di - minus_di) / adx | Irányossági index normalizálva ADX-szel |

---

## 3. Rolling stat-ok (67)

Négy indikátorra (RSI, MACD hist, CCI, MFI) × négy ablakra (5, 10, 20, 50) × négy stat:

| Sablon | Ablakok | Példa |
|--------|---------|-------|
| `{ind}_rmean_{w}` | 5, 10, 20, 50 | `rsi_rmean_20` |
| `{ind}_rstd_{w}` | 5, 10, 20, 50 | `macd_hist_rstd_10` |
| `{ind}_rmin_{w}` | 5, 10, 20, 50 | `cci_rmin_50` |
| `{ind}_rmax_{w}` | 5, 10, 20, 50 | `mfi_rmax_5` |

**ind** ∈ {rsi, macd_hist, cci, mfi} → 4 × 4 × 4 = **64 feature**

Plusz ATR relatív rolling mean:

| Feature | Mit mér |
|---------|---------|
| `atr_pct_rmean_10` | Volatilitás átlaga 10 gyertyán |
| `atr_pct_rmean_20` | Volatilitás átlaga 20 gyertyán |
| `atr_pct_rmean_50` | Volatilitás átlaga 50 gyertyán |

---

## 4. Lag + momentum feature-ök (16)

| Feature | Mit mér |
|---------|---------|
| `ret_lag_1` | Log-hozam 1 gyertyával ezelőtt |
| `ret_lag_2` | Log-hozam 2 gyertyával ezelőtt |
| `ret_lag_3` | Log-hozam 3 gyertyával ezelőtt |
| `ret_lag_5` | Log-hozam 5 gyertyával ezelőtt |
| `ret_lag_8` | Log-hozam 8 gyertyával ezelőtt (Fibonacci) |
| `ret_lag_13` | Log-hozam 13 gyertyával ezelőtt (Fibonacci) |
| `cum_ret_3` | Kumulált hozam 3 gyertyán |
| `cum_ret_5` | Kumulált hozam 5 gyertyán |
| `cum_ret_10` | Kumulált hozam 10 gyertyán |
| `cum_ret_20` | Kumulált hozam 20 gyertyán |
| `rsi_delta_3` | RSI változása 3 gyertyán (lendület üteme) |
| `rsi_delta_5` | RSI változása 5 gyertyán |
| `rsi_delta_10` | RSI változása 10 gyertyán |
| `obv_delta_5` | OBV változása 5 gyertyán (volumen lendület) |
| `obv_delta_20` | OBV változása 20 gyertyán |
| `obv_zscore` | OBV Z-score 20 gyertyás ablakban |

---

## 5. Cross-feature interakciók (9)

| Feature | Képlet | Mit mér |
|---------|--------|---------|
| `volume_ratio` | volume / rolling_mean(volume, 20) | Aktuális volumen az átlaghoz képest |
| `volume_x_ret` | volume_ratio × log_ret | Irányos volumen lendület |
| `rsi_x_vol` | rsi × volume_ratio | Oversold/overbought + volumen egyszerre |
| `adx_x_macd` | adx × sign(macd_hist) | Trend erőssége × irány kombinációja |
| `bb_width_delta` | bb_width_pct.diff(5) | BB összehúzódik/tágul-e? (squeeze/expansion) |
| `candle_body_pct` | \|close - open\| / (high - low) | Gyertya test/árnyék arány (0=doji, 1=marubozu) |
| `candle_direction` | sign(close - open) | Gyertya iránya (+1 bullish, -1 bearish) |
| `upper_wick_pct` | (high - max(open,close)) / (high - low) | Felső árnyék arány — elutasított emelkedés |
| `lower_wick_pct` | (min(open,close) - low) / (high - low) | Alsó árnyék arány — elutasított esés |

---

## 6. Volatilitás rezsim feature-ök (5)

| Feature | Képlet | Mit mér |
|---------|--------|---------|
| `realized_vol_5` | std(log_ret, 5) × √(252×24) | Annualizált realizált vol 5 gyertyán |
| `realized_vol_10` | std(log_ret, 10) × √(252×24) | Annualizált realizált vol 10 gyertyán |
| `realized_vol_20` | std(log_ret, 20) × √(252×24) | Annualizált realizált vol 20 gyertyán |
| `vol_ratio_5_20` | realized_vol_5 / realized_vol_20 | Rövid vs hosszú vol arány — rezsim jelzője |
| `parkinson_vol` | √(1/4ln2) × ln(high/low) | Parkinson-féle vol (high-low alapú, pontosabb) |

---

## 7. Orderbook proxy — OHLCV-alapú (3)

| Feature | Mit mér |
|---------|---------|
| `ob_imbalance_proxy` | Vételi/eladási nyomás becsülve gyertya pozícióból + volumenből |
| `ob_imbalance_proxy_ma_5` | OB imbalance proxy 5 gyertyás simítva |
| `ob_imbalance_proxy_ma_10` | OB imbalance proxy 10 gyertyás simítva |

*Élő módban ezek helyett valódi L2 orderbook adatot használ az `orderbook_features.py`.*

---

## 8. Funding rate feature-ök (5) \*

*Csak ha `funding_df` rendelkezésre áll (Bybit perp historikus adat)*

| Feature | Mit mér |
|---------|---------|
| `funding_rate` | Aktuális 8 óránkénti funding ráta |
| `funding_rate_ma_3` | Funding ráta 3 periódus simítva |
| `funding_rate_extreme` | 1 ha \|funding\| ≥ 0.1% (extrém zsúfolt pozíció) |
| `funding_rate_delta` | Funding ráta változása |
| `funding_annualized` | Éves szintre vetített funding (rate × 3 × 365) |

---

## 9. Open interest feature-ök (3) \*

*Csak ha `oi_df` rendelkezésre áll*

| Feature | Mit mér |
|---------|---------|
| `oi_norm` | OI / rolling_mean(OI, 20) — normalizált meggyőződés szint |
| `oi_delta_pct` | OI 3 periódus alatti %-os változása |
| `oi_price_conf` | oi_delta_pct × sign(log_ret) — OI és ár irányának egyezése |

---

## 10. Cross-timeframe feature-ök (4 × TF) \*

*Minden megadott magasabb timeframe-re (pl. "4h", "1d")*

| Sablon | Példa (4h TF-re) | Mit mér |
|--------|-----------------|---------|
| `rsi_{tf}` | `rsi_4h` | RSI a magasabb TF-en |
| `macd_hist_{tf}` | `macd_hist_4h` | MACD hist a magasabb TF-en |
| `atr_pct_{tf}` | `atr_pct_4h` | Relatív volatilitás magasabb TF-en |
| `rsi_div_{tf}` | `rsi_div_4h` | RSI divergencia (alap TF - magasabb TF) |

---

## 11. Makro feature-ök (9) \*

*`macro_history` DataFrame-ből (tanításhoz) vagy élő `CryptoDataSnapshot`-ból*

| Feature | Forrás | Mit mér |
|---------|--------|---------|
| `macro_sp500` | yfinance ^GSPC | SP500 szintje |
| `macro_dxy` | yfinance DX-Y.NYB | Dollár index szintje |
| `macro_vix` | yfinance ^VIX | Tőzsdei félelem index |
| `macro_gold` | yfinance GC=F | Arany ára |
| `macro_sp500_ret_5d` | SP500.pct_change(5) | SP500 5 napos hozam |
| `macro_sp500_ret_20d` | SP500.pct_change(20) | SP500 20 napos hozam |
| `macro_dxy_ret_5d` | DXY.pct_change(5) | Dollár 5 napos trend |
| `macro_vix_high` | VIX ≥ 30 | Tőzsdei pánik flag |
| `macro_vix_extreme` | VIX ≥ 40 | Extrém pánik flag |

---

## 12. On-chain feature-ök (3 × metrika) \*

*`onchain_history` dict-ből (pl. hash-rate, mempool-size, n-transactions)*

| Sablon | Példa | Mit mér |
|--------|-------|---------|
| `onchain_{metric}` | `onchain_hash_rate` | Nyers metrika értéke |
| `onchain_{metric}_ret_7d` | `onchain_hash_rate_ret_7d` | 7 napos változás |
| `onchain_{metric}_zscore_30` | `onchain_hash_rate_zscore_30` | Z-score 30 napos ablakban |

Ajánlott metrikák: `hash-rate`, `mempool-size`, `n-transactions`, `miners-revenue`
→ 3 feature × 4 metrika = **12 feature**

---

## 13. Likvidáció feature-ök (4) \*

| Feature | Mit mér |
|---------|---------|
| `liq_total_usd` | Összes likvidáció USD-ben az elmúlt 1 órában |
| `liq_ratio` | Long likvidáció / összes (0.5 = kiegyensúlyozott) |
| `liq_long_dom` | 1 ha liq_ratio ≥ 0.75 (long squeeze folyamatban) |
| `liq_short_dom` | 1 ha liq_ratio ≤ 0.25 (short squeeze folyamatban) |

---

## 14. Options feature-ök (4) \*

| Feature | Mit mér |
|---------|---------|
| `opt_put_call_ratio` | Put / Call open interest arány (>1.5 = félelem) |
| `opt_iv_atm` | At-the-money implied volatility (éves %) |
| `opt_iv_skew` | Put IV - Call IV (pozitív = downside-ot áraz a piac) |
| `opt_fear_flag` | 1 ha put_call_ratio ≥ 1.5 |

---

## 15. Hír sentiment (2) \*

| Feature | Mit mér |
|---------|---------|
| `news_sentiment` | CryptoPanic aggregált sentiment score (-1..+1) |
| `news_extreme_bull` | 1 ha sentiment ≥ 0.5 (eufória flag — kontrarian) |

---

## Összesítés

| Réteg | Feature-ök száma | Feltétel |
|-------|-----------------|----------|
| 1. Nyers indikátorok | 22 | mindig |
| 2. Normalizált pozíció | 13 | mindig |
| 3. Rolling stat-ok | 67 | mindig |
| 4. Lag + momentum | 16 | mindig |
| 5. Cross-feature | 9 | mindig |
| 6. Volatilitás rezsim | 5 | mindig |
| 7. Orderbook proxy | 3 | mindig |
| 8. Funding rate | 5 | funding_df megadva |
| 9. Open interest | 3 | oi_df megadva |
| 10. Cross-timeframe | 4 × TF | higher_tf_ohlcv megadva |
| 11. Makro | 9 | macro_history / crypto_snap |
| 12. On-chain | 3 × metrika | onchain_history / crypto_snap |
| 13. Likvidáció | 4 | crypto_snap |
| 14. Options | 4 | crypto_snap |
| 15. Hír sentiment | 2 | crypto_snap |
| **Alap (1–7)** | **135** | **mindig** |
| **Teljes (minden forrással, 2 TF + 4 on-chain metrika)** | **~172** | **ha minden adat elérhető** |
