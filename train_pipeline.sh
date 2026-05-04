#!/bin/bash
# =============================================================================
# Teljes tanítási pipeline
#
# Mit csinál:
#   1. Historikus OHLCV letöltés (Binance — 5 év, 1h + 4h)
#   2. Makro adat letöltés (SP500, DXY, VIX, Gold — yfinance)
#   3. On-chain adat letöltés (hash rate, mempool — blockchain.com)
#   4. ML modell tanítás minden symbol/timeframe kombinációra
#   5. Eredmény összesítés
#
# Futtatás:
#   screen -S training
#   bash train_pipeline.sh 2>&1 | tee logs/trainer.log
#   Ctrl+A, D  (háttérbe)
# =============================================================================

set -e

PROJECT_DIR="/root/bybit-trader"
ENV="/root/trader-env/bin/activate"
LOG_DIR="$PROJECT_DIR/logs"
DATA_DIR="$PROJECT_DIR/data"
MODEL_DIR="$PROJECT_DIR/models"

# Konfigurálható paraméterek
YEARS=5
TIMEFRAMES="1h 4h"
SYMBOLS="BTC/USDT ETH/USDT BNB/USDT SOL/USDT XRP/USDT"
MAX_HOLDING=20
FOLDS=5
N_ESTIMATORS=300

# =============================================================================
source "$ENV"
cd "$PROJECT_DIR"
source .env 2>/dev/null || true

mkdir -p "$LOG_DIR" "$MODEL_DIR"
START_TIME=$(date +%s)

echo "=============================================="
echo " Bybit Trader — Training Pipeline"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
echo " Symbols:    $SYMBOLS"
echo " Timeframes: $TIMEFRAMES"
echo " History:    $YEARS év"
echo " Folds:      $FOLDS"
echo "=============================================="

# =============================================================================
# 1. HISTORIKUS OHLCV LETÖLTÉS
# =============================================================================
echo ""
echo "[1/4] OHLCV letöltés (Binance)..."
echo "  Figyelem: Binance EU/UK-ból blokkolva lehet — VPS Frankfurt-ból OK"

for TF in $TIMEFRAMES; do
    echo "  Timeframe: $TF"
    python fetch_history.py \
        --exchange binance \
        --symbols $SYMBOLS \
        --timeframes "$TF" \
        --years "$YEARS" \
        --out-dir "$DATA_DIR" \
        2>&1 | sed 's/^/    /'
done

echo "  Letöltött fájlok:"
ls -lh "$DATA_DIR"/*.csv 2>/dev/null | awk '{print "    " $5 "  " $9}' || echo "    (nincs CSV)"

# =============================================================================
# 2. MAKRO ADAT LETÖLTÉS
# =============================================================================
echo ""
echo "[2/4] Makro adat letöltés (yfinance)..."

python3 - << 'PYEOF'
import sys
sys.path.insert(0, ".")
from crypto_data import fetch_macro_history
import os

print("  SP500, DXY, VIX, Gold letöltése...")
df = fetch_macro_history(years=6.0)   # kicsit több mint az OHLCV
if df.empty:
    print("  HIBA: makro adat nem érhető el")
    sys.exit(0)

out = "data/macro_history.csv"
df.to_csv(out)
print(f"  Mentve: {out}  ({len(df)} sor)")
PYEOF

# =============================================================================
# 3. ON-CHAIN ADAT LETÖLTÉS
# =============================================================================
echo ""
echo "[3/4] On-chain adat letöltés (blockchain.com)..."

python3 - << 'PYEOF'
import sys
sys.path.insert(0, ".")
from crypto_data import fetch_onchain_history

metrics = {
    "hash_rate":    ("hash-rate",       "1year"),
    "mempool":      ("mempool-size",    "1year"),
    "tx_count":     ("n-transactions",  "1year"),
    "miner_rev":    ("miners-revenue",  "1year"),
}

for name, (metric, span) in metrics.items():
    df = fetch_onchain_history(metric, timespan=span)
    if df.empty:
        print(f"  {name}: nem érhető el (kihagyva)")
        continue
    out = f"data/onchain_{name}.csv"
    df.to_csv(out)
    print(f"  {name}: {len(df)} sor → {out}")
PYEOF

# =============================================================================
# 4. ML MODELL TANÍTÁS
# =============================================================================
echo ""
echo "[4/4] ML modell tanítás..."

TRAINED=0
FAILED=0

for TF in $TIMEFRAMES; do
    for SYM in $SYMBOLS; do
        # CSV fájlnév generálás (fetch_history.py konvenció)
        SAFE_SYM=$(echo "$SYM" | tr '/' '_')
        CSV="$DATA_DIR/${SAFE_SYM}_${TF}_${YEARS}y_binance.csv"
        MODEL="$MODEL_DIR/${SAFE_SYM}_${TF}.pkl"

        if [ ! -f "$CSV" ]; then
            echo "  KIHAGYVA (nincs CSV): $CSV"
            FAILED=$((FAILED + 1))
            continue
        fi

        echo ""
        echo "  ─── $SYM  $TF ───"
        echo "  CSV:   $CSV"
        echo "  Model: $MODEL"

        python main.py ml-train \
            --csv        "$CSV" \
            --model-out  "$MODEL" \
            --symbol     "$SYM" \
            --timeframe  "$TF" \
            --folds      "$FOLDS" \
            --max-holding "$MAX_HOLDING" \
            --n-estimators "$N_ESTIMATORS" \
            2>&1 | grep -E "(Fold|accuracy|feature|mentve|HIBA|Train)" | sed 's/^/    /'

        if [ -f "$MODEL" ]; then
            SIZE=$(du -sh "$MODEL" | cut -f1)
            echo "  ✓ Modell kész: $SIZE"
            TRAINED=$((TRAINED + 1))
        else
            echo "  ✗ Modell nem jött létre"
            FAILED=$((FAILED + 1))
        fi
    done
done

# =============================================================================
# ÖSSZESÍTÉS
# =============================================================================
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
MINUTES=$(( ELAPSED / 60 ))
SECONDS=$(( ELAPSED % 60 ))

echo ""
echo "=============================================="
echo " KÉSZ — $(date '+%Y-%m-%d %H:%M:%S')"
echo " Futási idő: ${MINUTES}m ${SECONDS}s"
echo "=============================================="
echo " Betanított modellek: $TRAINED"
echo " Sikertelen:          $FAILED"
echo ""
echo " Modellek helye: $MODEL_DIR"
ls -lh "$MODEL_DIR"/*.pkl 2>/dev/null | awk '{print "   " $5 "  " $9}' || echo "   (nincs modell)"
echo ""
echo " Következő lépés:"
echo "   scp -r root@<VPS_IP>:$MODEL_DIR ./models"
echo "   python main.py paper --symbol BTC/USDT --timeframe 1h"
echo "=============================================="
