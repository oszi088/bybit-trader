# A rendszer működése — részletes architektúra leírás

## Áttekintés

A rendszer két egymásra épülő döntési rétegből áll:

1. **Elsődleges jelrendszer** — 13 technikai indikátor súlyozott szavazása
2. **Meta-label ML réteg** — XGBoost modell, amely azt tanulja meg, hogy az elsődleges jelrendszer *mikor téved*

A két réteg kombinációja pontosabb döntéseket eredményez, mint bármelyik önmagában.

---

## 1. Adatbevitel

Minden döntési ciklus három adat-réteget olvas be:

```
┌─────────────────────────────────────────────────────────┐
│                      ADAT RÉTEGEK                       │
├──────────────────┬──────────────────┬───────────────────┤
│   OHLCV (ár)     │  Alternatív adat │  Orderbook        │
│                  │                  │                   │
│  Bybit spot      │  • SP500/VIX/DXY │  Bybit L2         │
│  1m – 1M         │  • Hash rate     │  bid/ask mélység  │
│  történeti CSV   │  • Mempool       │                   │
│  vagy élő CCXT   │  • Likvidációk   │  (backtestben     │
│                  │  • Put/Call arány│   OHLCV-proxy)    │
│                  │  • Hír sentiment │                   │
└──────────────────┴──────────────────┴───────────────────┘
```

**Backtestben:** az alt-data historikus CSV-ekből töltődik (yfinance, blockchain.com), az orderbook OHLCV-alapú közelítéssel helyettesítődik.

**Élő módban:** minden forrás párhuzamos API-hívással frissül, cache-eléssel (funding: 1h TTL, makro: 1h, orderbook: valós idejű).

---

## 2. Feature engineering — 132 feature, 15 réteg

Az nyers adatokból a `ml_features_v2.py` összesen ~132 feature-t számít:

| # | Réteg | Feature-ök (~db) | Példák |
|---|-------|-----------------|--------|
| 1 | Nyers indikátor értékek | 22 | RSI, MACD hist, ADX, Stochastic %K/%D |
| 2 | Normalizált pozíció | 10 | BB %B, MA-távolságok, ATR relatív |
| 3 | Rolling stat-ok | 20 | RSI rolling mean/std 5/10/20/50 gyertyán |
| 4 | Lag + momentum | 15 | Log-hozam lag-ok (1,2,3,5,8,13), kumulált hozamok |
| 5 | Cross-feature interakciók | 10 | RSI × volume, ADX × MACD irány, gyertya body % |
| 6 | Volatilitás rezsim | 8 | Realized vol, Parkinson vol, vol-arány |
| 7 | Orderbook proxy | 4 | OB imbalance (OHLCV-alapú), rolling átlaga |
| 8 | Funding rate | 5 | Aktuális ráta, 3 periódus átlag, extrém flag |
| 9 | Open interest | 4 | OI normalizált, 3 napos változás, OI×ár konfidencia |
| 10 | Cross-timeframe | 10 | RSI/MACD/ATR magasabb TF-ről (4h, 1d), divergenciák |
| 11 | Makro (SP500/DXY/VIX) | 8 | SP500 1d/7d hozam, DXY trend, VIX szint + flag-ek |
| 12 | On-chain | 6 | Hash rate, hash ret 7d, mempool, tx count, Z-score-ok |
| 13 | Likvidációk | 4 | Összes liq USD, long/short arány, squeeze flag-ek |
| 14 | Options | 4 | Put/Call arány, IV ATM, IV skew, félelem flag |
| 15 | Hír sentiment | 2 | CryptoPanic score, extrém bullish flag |

**Fontos:** minden feature kizárólag múltbeli adatból számítódik. Nincs lookahead bias.

---

## 3. Elsődleges jelrendszer

A `TradingAgent` 13 indikátort futtat, mindegyik `{-1, 0, +1}` szavazatot ad:

```
OHLCV
  │
  ├─► indicators.py ──► 13 indikátor értéke
  │                              │
  │                    signals.py konvertálja
  │                              │
  │         ┌────────────────────┼────────────────────┐
  │         │                   │                    │
  │    Trend (4db)        Momentum (3db)       Volumen (3db)
  │    SMA cross          RSI                  OBV
  │    EMA cross          Stochastic           VWAP
  │    MACD               CCI                  MFI
  │    ADX/±DI
  │         │                   │                    │
  │    Volatilitás (1db)   Makro (1db)        Hosszú táv (2db)
  │    Bollinger Bands     Fear & Greed        Golden/Death cross
  │                                            Long trend (SMA50/200)
  │
  └─► Súlyozott szavazás ──► score ∈ [-1, +1]
```

### Rezsim-aware súlyozás

A piac aktuális állapotát a `RegimeDetector` azonosítja az ADX alapján:

| Rezsim | Feltétel | Domináns indikátorok |
|--------|----------|---------------------|
| **Trend** | ADX > 25 | MACD, EMA cross, SMA cross (2.0x súly) |
| **Range** | ADX < 18 | Bollinger, RSI, Stochastic (2.0x súly) |
| **Neutral** | 18 ≤ ADX ≤ 25 | Default súlyozás |
| **Scalping** | 1m/3m/5m TF | Rövidebb periódusok, ±0.55 küszöb |

### Multi-TimeFrame megerősítés

Az alacsonyabb TF döntését 5 magasabb TF SMA-cross trendje befolyásolja:

```
Alap döntés (pl. 1h) ──► score_alap
                                 +
MTF elemzés:
  6h  (súly: 0.5)  ──► trend jel
  12h (súly: 0.7)  ──► trend jel
  1d  (súly: 1.0)  ──► trend jel    ──► composite_score (±1)
  1w  (súly: 1.5)  ──► trend jel         × mtf_weight (1.5)
  1M  (súly: 2.0)  ──► trend jel
                                 │
                    score_végleges = (score_alap × Σw_ind + composite × 1.5)
                                     ────────────────────────────────────────
                                              Σ(összes súly)
```

**Gate mód:** ha az MTF erősen ellentmond az alap jelnek (threshold: 0.3), a döntés HOLD-ra vált.

### Döntési küszöb

```
score ≥ +0.40  →  BUY
score ≤ -0.40  →  SELL
egyébként      →  HOLD
```

---

## 4. Triple Barrier Labeling (tanításhoz)

A hagyományos "következő gyertya iránya" label félrevezető — nem veszi figyelembe a valódi kockázatot. Helyette dynamic barrier-eket használunk:

```
         TP barrier: entry + atr_tp_mult × ATR  ──► label = +1
              ▲
    entry ────●──────────────────────► idő
              ▼
         SL barrier: entry - atr_stop_mult × ATR ──► label = -1

    Ha egyik sem érintődik max_holding gyertyán belül  ──► label = 0
```

**Eredmény:** a label azt méri, hogy a valódi kockázatkezelési szabályok (ATR stop + TP) mellett nyereséges lett volna-e a belépés — nem csupán azt, hogy az ár emelkedett-e.

---

## 5. Meta-label ML réteg

### Az alapötlet

Az ML **nem** azt tanulja, merre megy az ár (zajosan és nehezen tanítható). Helyette azt tanulja: **az elsődleges jelrendszer mikor ad megbízható jelet.**

```
Elsődleges jel: "BUY"
                    │
                    ▼
    XGBoost meta-labeler
    132 feature alapján:
    "P(ez a BUY jel helyes) = 0.71"
                    │
         ┌──────────┴──────────┐
         │ p ≥ 0.55            │ p < 0.55
         ▼                     ▼
     tényleg BUY           HOLD (kihagyjuk)
     bet_size = 0.42       (nem megbízható)
```

### Meta-label definíció

```
meta_label = 1  ha:  primary_signal != 0  ÉS  sign(primary) == sign(triple_barrier_label)
meta_label = 0  egyébként
```

### Purged Walk-Forward Cross-Validation

A klasszikus train/test split a kereskedési adatokon megbízhatatlan, mert a nyitott pozíciók "átlógnak" a határon (label szivárgás). A megoldás:

```
Teljes idósor:
│◄──────── Train ────────►│◄ embargo ►│◄──── Test ────►│
│                         │           │                │
│  Pozíció nyílik itt     │  tiltott  │  Pozíció zár   │
│  ──────────────────────►│   zóna    │  itt           │
│                         │           │                │

Az embargo zóna kizárja a train-ből azokat a sorokat,
ahol a nyitott pozíció label-je "belenyúlik" a test-be.
```

**5 fold walk-forward**, minden fold csak az idősorban korábbi adatokon tanul.

### Bet sizing (Kelly-proxy)

```
P(jel helyes) → pozícióméret szorzó

p = 0.50  →  bet = 0.00  (semleges, nem lépünk be)
p = 0.62  →  bet = 0.25
p = 0.75  →  bet = 0.50
p = 1.00  →  bet = 1.00  (teljes konfidencia)

bet_size = max(0, 2 × (p - 0.5))
```

---

## 6. Kockázatkezelés

Minden belépési döntés átmegy a `RiskManager` szűrőin:

```
Döntés: BUY
    │
    ├── halted? ──────────────────────────────► BLOKKOLVA
    │
    ├── order_value > max_order_usd? ──────────► BLOKKOLVA
    │
    ├── ATR/ár > max_atr_pct? ─────────────────► BLOKKOLVA (túl volatilis)
    │
    ├── napi PnL < -daily_loss_limit? ──────────► KILL SWITCH
    │
    ├── drawdown > max_drawdown_pct? ──────────► KILL SWITCH
    │
    └── ML bet_size szorzó ──────────────────────► pozíció méret ×0..1
```

### Stop-ok

```
Belépés @ entry_price
    │
    ├── Stop-loss:    entry - atr_stop_mult × ATR  (default: 3×ATR)
    ├── Take-profit:  entry + atr_tp_mult × ATR    (default: 5×ATR)
    └── Trailing stop: max_ár - trailing_atr_mult × ATR  (scalping módban)
```

---

## 7. Teljes döntési folyamat (egyetlen gyertya)

```
Új gyertya érkezik
        │
        ▼
   OHLCV frissítés
        │
        ├──► Indikátorok számítása (indicators.py)
        ├──► Feature matrix frissítése (ml_features_v2.py)
        ├──► Alt-data lekérés cache-ből (crypto_data.py)
        │
        ▼
   Rezsim detektálás (regime.py)
        │
        ▼
   13 indikátor szavazás (signals.py + agent.py)
        │
        ▼
   MTF megerősítés (mtf.py)
        │
        ▼
   Súlyozott aggregáció → score ∈ [-1, +1]
        │
        ▼
   Elsődleges döntés: BUY / SELL / HOLD
        │
        ▼ (ha BUY vagy SELL)
   XGBoost meta-labeler → P(jel helyes)
        │
        ├── p < 0.55 → HOLD (ML kiszűri)
        │
        └── p ≥ 0.55 → döntés megerősítve
                │
                ▼
        RiskManager szűrők
                │
                ▼
        Pozíció méretezés (Kelly × bet_size)
                │
                ▼
        Megbízás küldés (Broker)
                │
                ▼
        SQLite log + Telegram értesítés
```

---

## 8. Tanítási munkafolyamat

```bash
# 1. Historikus adat letöltése (Binance, VPS-ről)
python fetch_history.py --exchange binance \
    --symbols BTC/USDT ETH/USDT \
    --timeframes 1h 4h \
    --years 5

# 2. ML modell tanítása
python main.py ml-train \
    --csv data/BTC_USDT_1h_5y_binance.csv \
    --model-out models/btc_1h.pkl \
    --folds 5 \
    --max-holding 20

# 3. Eredmény értékelése
# A kimenet tartalmazza:
#   - Fold-onkénti accuracy
#   - Top 20 legfontosabb feature (MDI)
#   - Pozitív arány (hány jelet hagyott át)

# 4. Paper trading ML szűrővel
python main.py paper --symbol BTC/USDT --timeframe 1h

# 5. Walkforward stabilitás ellenőrzés
python main.py walkforward --csv data/BTC_USDT_1h_5y_binance.csv
```

---

## 9. Fájlstruktúra

```
Alap jelrendszer:
  config.py          — paraméterek, súlyok, rezsim config
  indicators.py      — 13 indikátor pandas implementációja
  signals.py         — indikátor → {-1, 0, +1} konverzió
  regime.py          — trend/range/neutral rezsim detektor
  agent.py           — súlyozott szavazás + MTF + ML integráció
  mtf.py             — Multi-TimeFrame elemzés

ML réteg:
  triple_barrier.py  — ATR-alapú labeling (TP/SL/időlimit)
  ml_features.py     — ~50 feature (alap verzió)
  ml_features_v2.py  — ~132 feature (teljes verzió)
  ml_model.py        — XGBoost meta-labeler, purged walk-forward CV
  ml_train.py        — tanítási pipeline

Alternatív adat:
  alt_data.py        — Bybit funding rate, OI, long/short, BTC dominance
  crypto_data.py     — SP500/VIX/DXY, hash rate, likvidációk, options, hírek
  orderbook_features.py — L2 orderbook mikrostruktúra

Kereskedési motor:
  brokers.py         — PaperBroker + BybitBroker (order visszaigazolással)
  risk_manager.py    — plafon, daily kill switch, drawdown védelem
  trader.py          — élő/paper loop, ATR stop, trailing, watchdog
  portfolio.py       — multi-symbol manager, közös kockázat

Infrastruktúra:
  data_source.py     — CSV + CCXT adatforrás
  fetch_history.py   — historikus adat letöltő (Bybit/Binance)
  optimizer.py       — walk-forward grid search, overfit szűrővel
  backtest.py        — slippage-es backteszt, walk-forward
  db.py              — SQLite trade log
  notify.py          — Telegram értesítés
  coins.py           — top 20 USDT pár lista
  main.py            — CLI belépőpont
```

---

## 10. Ami megkülönbözteti az egyszerű indikátor-alapú rendszerektől

| Jellemző | Egyszerű rendszer | Ez a rendszer |
|----------|-------------------|---------------|
| Label | Következő gyertya iránya | Triple barrier (TP/SL/időlimit) |
| ML cél | Irány megjóslása | Mikor megbízható a jel? |
| CV módszer | Random split | Purged walk-forward + embargo |
| Pozíció méret | Fix | Kelly-proxy × ML konfidencia |
| Adatforrás | OHLCV | OHLCV + funding + OI + makro + on-chain + options |
| Rezsim | Fix súlyok | Trend/Range/Scalping-aware súlyok |
| Stop | Fix % | ATR-adaptív (volatilitással skálázódik) |
| Overfit védelem | Nincs | OOS szűrő + embargo + konzervatív hiperparaméterek |
