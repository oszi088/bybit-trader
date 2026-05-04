# Bybit Trader — Projekt Szabályok

Ezt a fájlt Claude Code minden session elején olvassa. Tartalmazza az
architektúra elveit, fejlesztési szabályokat és a piac-adaptációs
rendszer működési logikáját.

---

## Projekt Architektúra

```
market_cycle.py      → Ciklus detektálás (ACCUMULATION → BULL_LATE → BEAR stb.)
adaptive_strategy.py → Ciklusonkénti paraméterek (stop, TP, pozícióméret)
agent.py             → TradingAgent: jelzések + ML + ciklus integráció
ml_model.py          → XGBoost meta-label modell (amikor kell kereskedni)
ml_features_v2.py    → 135+ feature mátrix (15 rétegben)
triple_barrier.py    → Lopez de Prado triple barrier labeling
ml_train.py          → Teljes tanítási pipeline
crypto_data.py       → Makro + on-chain + likvidáció + opciós adatok
alt_data.py          → Funding rate, OI, long/short arány, BTC dominancia
fear_greed.py        → Fear & Greed index (alternative.me)
signals.py           → Technikai jelzések (RSI, MACD, BB stb.)
indicators.py        → OHLCV → indikátorok számítása
regime.py            → Rövid távú piaci rezsim (Trend/Range/Scalping)
config.py            → TradingConfig (minden paraméter egy helyen)
brokers.py           → Bybit API (CCXT), order végrehajtás
notify.py            → Telegram értesítések
```

---

## Döntéshozatal folyamata (minden gyertya)

```
1. market_cycle.detect()     → MarketCycle (pl. BULL_MID)
2. adaptive_strategy.get_params(cycle) → CycleRegimeParams
3. regime.detect_regime()    → rövid távú rezsim (Trend/Range stb.)
4. signals.signal_*()        → technikai jelzések szavazása
5. ml_model.predict()        → meta-label konfidencia (p, bet_size)
6. TradingAgent._aggregate() → súlyozott összesítés
7. Döntés: BUY / SELL / HOLD
   - Ha cycle nem engedi az irányt → HOLD kényszer
   - Ha ML p < min_ml_prob      → HOLD kényszer
   - Ha score < buy_threshold   → HOLD
```

---

## Piac-Adaptáció Szabályai

### Soha ne hardkódolj piaci feltételezést
A ciklus-detektáló (`market_cycle.py`) automatikusan meghatározza,
hol tartunk. A stratégia az `adaptive_strategy.CYCLE_PARAMS` alapján
alkalmazkodik. Ha új piaci viselkedést észlelsz, a CYCLE_PROFILES-t
kell frissíteni — nem az agent.py-t.

### Ciklus hozzáadása
1. Adj hozzá elemet a `MarketCycle` enum-hoz
2. Töltsd ki a `CYCLE_DURATIONS`, `NEXT_CYCLE`, `CYCLE_PROFILES` dict-eket
3. Adj hozzá `CycleRegimeParams` bejegyzést az `adaptive_strategy.CYCLE_PARAMS`-ba
4. Futtasd: `python -c "from market_cycle import *; print('OK')"`

### Ciklus paraméter módosítása
Csak `adaptive_strategy.py` CYCLE_PARAMS-ban módosíts.
Soha ne írj ciklus-specifikus if-eket az agent.py-ba.

### Bull run időtartam becslése
A `CycleState.days_remaining_est` = historikus átlag − eltelt napok.
Ha a valós piac eltér, a `CYCLE_DURATIONS` dict frissítendő.
Jelenlegi historikus átlagok (bitcoin 2013–2024 adatok alapján):
  BULL_EARLY:   75 nap átlag (30–120)
  BULL_MID:    150 nap átlag (90–240)
  BULL_LATE:    45 nap átlag (14–90)
  Teljes bull ciklus: ~270 nap átlag

---

## Fejlesztési Szabályok

### Feature hozzáadása
- Minden feature a `ml_features_v2.py` `build_feature_matrix_v2()` függvénybe kerül
- Feature neve: `snake_case`, legyen önleíró (pl. `btc_dom_7d_chg`)
- A feature opcionális adatforrástól függhet → `if funding_df is not None:` pattern
- Frissítsd a FEATURES.md-t, ha új feature réteget adsz hozzá

### Model változtatás
- Az XGBoost paraméterei `ml_train.py`-ban módosíthatók (`--n-estimators`, stb.)
- Ne változtasd a triple barrier logikát tanítás közben — inkonzisztens labelt ad
- A purged K-fold embargo periódusa = `max_holding_bars` gyertya

### Kockázatkezelés — nem tárgyalható szabályok
- MAX pozíció per trade: `CycleRegimeParams.max_position_pct` × portfólió
- Stop loss MINDEN pozícióhoz kötelező (`atr_stop_mult × ATR`)
- Kill switch: ha `risk_manager.py` triggerel, az agent NEM írhatja felül
- RISK_OFF ciklusban: `allow_long = False` mindig marad

### API kulcsok és titkok
- `.env` fájl SOHA nem kerül a repóba (`.gitignore`-ban van)
- API kulcsok csak `.env`-ből olvashatók (`os.getenv()`)
- Testnet módban fejlessz, live módba csak tesztelt kód kerülhet

### Kód stílus
- Type hints mindenhol (dataclass, return type)
- Docstring: mit csinál a függvény, mik a bemenetek, mi a kimenet
- Logger: `logging.getLogger(__name__)` — ne használj print()-et
- Kivételkezelés: mindig log a hibát, soha ne csökkentsd a positiont
  csendben hiba esetén

### Tesztelés
- Minden új feature: `python -c "from ml_features_v2 import *"` ellenőrzés
- Backtest ELŐTT kell futtatni új stratégián: `python main.py backtest ...`
- Paper trading minimum 2 hét live adaton, mielőtt live-ra váltasz

---

## VPS Deployment

```bash
# Feltöltés és tanítás indítása
bash deploy.sh <VPS_IP>

# Log követés
ssh root@<VPS_IP> 'tail -f /root/bybit-trader/logs/trainer.log'

# Modellek letöltése
scp -r root@<VPS_IP>:/root/bybit-trader/models ./models
```

A VPS-en a `.env` fájlt kézzel kell kitölteni (Bybit API kulcsok).
A `setup_vps.sh` létrehozza a sablont.

---

## Adatforrások prioritása

| Forrás | Valós időben | Historikus | Ingyenes |
|--------|-------------|-----------|---------|
| Bybit OHLCV | ✓ | ✓ (1-2 év) | ✓ |
| Binance OHLCV | ✓ | ✓ (5+ év) | ✓ |
| Fear & Greed | ✓ | ✗ | ✓ |
| Funding Rate (Bybit) | ✓ | ✓ | ✓ |
| BTC Dominance (CoinGecko) | ✓ | ✗ | ✓ |
| Makro (yfinance) | ✓ | ✓ (20+ év) | ✓ |
| On-chain (blockchain.com) | ✓ | ✓ | ✓ |
| VIX | ✓ | ✓ | ✓ |

Ha egy forrás nem elérhető, a rendszer fallback értékekkel dolgozik
(soha nem áll le csak azért, mert egy API nem válaszol).

---

## Ismert Korlátok

- **Fear & Greed historikus adat**: nem elérhető — backtestben az árfolyam
  momentum-ból becsüljük (75 felett = extreme greed proxy)
- **On-chain NUPL/MVRV**: pontos adat csak fizetős API-n — proxyt használunk
  (close vs 365MA arány)
- **Funding rate backtest**: Bybit csak ~2 évet ad vissza — régebbi adatnál
  proxy (open interest változásából becsüljük)
- **ML modell drift**: 6 havonta újra kell tanítani, ahogy a piac viselkedése
  változik. A `train_pipeline.sh` erre való.
