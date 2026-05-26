import os, json, statistics
from collections import deque

MAX_HISTORY  = 200
HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history_cache.json")

class ProMaxPredictor:
    """
    Ensemble of 5 algorithms:
      1. SMA  - Simple Moving Average crossover
      2. EMA  - Exponential Moving Average (properly seeded, no double-count)
      3. Streak Detector
      4. Frequency Analyzer
      5. Volatility Scorer
    """

    def __init__(self):
        self.history   = deque(maxlen=MAX_HISTORY)
        self.round_num = 0

    def add(self, value: float):
        self.history.append(float(value))
        self.round_num += 1

    # ------------------------------------------------------------------ persist
    def save_history(self):
        """Persist history to disk so bot can resume after restart."""
        try:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(
                    {"history": list(self.history), "round_num": self.round_num}, f
                )
        except Exception as e:
            print(f"[PREDICTOR] Could not save history: {e}")

    def load_history(self):
        """Load persisted history from disk on startup."""
        if not os.path.exists(HISTORY_FILE):
            return
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            saved = data.get("history", [])
            self.round_num = data.get("round_num", 0)
            # Validate each value before loading
            for v in saved:
                fv = float(v)
                if 1.0 <= fv <= 200.0:
                    self.history.append(fv)
            print(
                f"[PREDICTOR] Loaded {len(self.history)} valid rounds "
                f"from cache (Round #{self.round_num})"
            )
        except Exception as e:
            print(f"[PREDICTOR] Could not load history: {e}")

    # ------------------------------------------------------------------ signals
    def _sma_signal(self, h):
        if len(h) < 5:
            return 0, "Not enough data for SMA"
        sma5  = statistics.mean(h[-5:])
        sma20 = statistics.mean(h[-20:]) if len(h) >= 20 else statistics.mean(h)
        if sma5 > sma20 * 1.1:
            return 2, f"SMA bullish: sma5={sma5:.2f} > sma20={sma20:.2f}"
        elif sma5 < sma20 * 0.9:
            return -2, f"SMA bearish: sma5={sma5:.2f} < sma20={sma20:.2f}"
        return 0, f"SMA neutral: sma5={sma5:.2f} vs sma20={sma20:.2f}"

    def _ema_signal(self, h):
        """EMA: seed from oldest value, iterate forward. No double-counting."""
        if len(h) < 3:
            return 0, "Not enough data for EMA"
        k   = 2 / (min(len(h), 10) + 1)
        ema = h[0]               # seed: oldest
        for v in h[1:]:          # iterate from second value forward (fixed)
            ema = v * k + ema * (1 - k)
        last = h[-1]
        if last > ema * 1.15:
            return 2, f"EMA bullish: last={last:.2f} > ema={ema:.2f}"
        elif last < ema * 0.85:
            return -2, f"EMA bearish: last={last:.2f} < ema={ema:.2f}"
        return 0, f"EMA neutral: ema={ema:.2f}"

    def _streak_signal(self, h):
        if len(h) < 3:
            return 0, "Not enough data for Streak"
        recent = list(reversed(h[-8:]))
        low_streak  = 0
        high_streak = 0
        for v in recent:
            if v < 2.0:
                low_streak += 1
            else:
                break
        for v in recent:
            if v >= 2.0:
                high_streak += 1
            else:
                break
        if low_streak >= 4:
            return 3, f"Low streak: {low_streak} consecutive crashes (<2x) - reversal likely"
        elif low_streak >= 3:
            return 2, f"Low streak: {low_streak} reds in a row"
        elif low_streak >= 2:
            return 1, f"Low streak: {low_streak} reds"
        elif high_streak >= 4:
            return -2, f"High streak: {high_streak} greens - crash likely"
        elif high_streak >= 3:
            return -1, f"High streak: {high_streak} greens"
        return 0, "No significant streak"

    def _frequency_signal(self, h):
        if len(h) < 10:
            return 0, "Not enough data for Frequency"
        n  = len(h)
        b1 = sum(1 for x in h if x < 1.5)       / n
        b2 = sum(1 for x in h if 1.5 <= x < 2)  / n
        b3 = sum(1 for x in h if 2 <= x < 5)    / n
        b4 = sum(1 for x in h if x >= 5)        / n
        crash_rate = b1 + b2
        if crash_rate > 0.7:
            return -1, f"High crash freq: {crash_rate*100:.0f}% under 2x"
        elif crash_rate < 0.4:
            return 2, f"Low crash freq: {crash_rate*100:.0f}% under 2x"
        elif b4 > 0.2:
            return 1, f"High 5x+ freq: {b4*100:.0f}%"
        return 0, f"Normal distribution: crash={crash_rate*100:.0f}%"

    def _volatility_signal(self, h):
        if len(h) < 5:
            return 0, "Not enough data for Volatility"
        window = h[-10:] if len(h) >= 10 else h
        # Need at least 2 values for stdev
        if len(window) < 2:
            return 0, "Not enough data for Volatility"
        std  = statistics.stdev(window)
        avg  = statistics.mean(window)
        last = h[-1]
        if last > avg + 2 * std:
            return -3, f"Extreme outlier high: {last:.2f}x (crash likely next)"
        elif last < avg - std and last < 1.5:
            return 2, f"Abnormally low round: {last:.2f}x (bounce possible)"
        return 0, f"Volatility normal: std={std:.2f}"

    # ------------------------------------------------------------------ predict
    def predict(self):
        h = list(self.history)
        n = len(h)

        if n < 3:
            return {
                "action":     "WATCH",
                "cashout":    "-",
                "confidence": 0,
                "conf_label": "Collecting data...",
                "reason":     f"Need more data ({n}/10 rounds collected)",
                "signal":     "neutral",
                "votes":      {},
                "stats":      self._stats(h),
                "total_score": 0,
            }

        s1, r1 = self._sma_signal(h)
        s2, r2 = self._ema_signal(h)
        s3, r3 = self._streak_signal(h)
        s4, r4 = self._frequency_signal(h)
        s5, r5 = self._volatility_signal(h)

        total        = s1 + s2 + s3 + s4 + s5
        max_possible = 12  # 2+2+3+2+3

        raw_conf   = (total / max_possible) * 50 + 50
        confidence = max(10, min(90, int(raw_conf)))

        if total >= 4:
            action, cashout, signal = "BET ??", "2.00x", "bet"
            conf_label = "Strong Buy" if total >= 6 else "Buy"
        elif total >= 1:
            action, cashout, signal = "BET SAFE ??", "1.50x", "safe"
            conf_label = "Weak Buy"
        elif total <= -3:
            action, cashout, signal = "SKIP ??", "-", "skip"
            conf_label = "Strong Skip"
            confidence = max(10, min(90, int((-total / max_possible) * 50 + 50)))
        elif total <= -1:
            action, cashout, signal = "WAIT ??", "1.50x", "wait"
            conf_label = "Caution"
        else:
            action, cashout, signal = "BET ??", "2.00x", "neutral"
            conf_label = "Neutral"
            confidence = 50

        # Pick most informative reason
        priority_keys = ["streak", "extreme", "abnormal", "bullish", "bearish", "crash"]
        all_reasons   = [r3, r5, r1, r2, r4]
        reason = next(
            (r for r in all_reasons if any(k in r.lower() for k in priority_keys)),
            all_reasons[0],
        )

        return {
            "action":     action,
            "cashout":    cashout,
            "confidence": confidence,
            "conf_label": conf_label,
            "reason":     reason,
            "signal":     signal,
            "votes": {
                "SMA":        {"score": s1, "reason": r1},
                "EMA":        {"score": s2, "reason": r2},
                "Streak":     {"score": s3, "reason": r3},
                "Frequency":  {"score": s4, "reason": r4},
                "Volatility": {"score": s5, "reason": r5},
            },
            "stats":       self._stats(h),
            "total_score": total,
        }

    # ------------------------------------------------------------------ stats
    def _stats(self, h):
        n = len(h)
        if n == 0:
            return {
                "total": 0, "avg_all": 0, "avg_10": 0, "avg_5": 0,
                "crash_rate": 0, "b1": 0, "b2": 0, "b3": 0, "b4": 0,
                "max_mult": 0, "min_mult": 0,
            }
        recent10 = h[-10:] if n >= 10 else h
        recent5  = h[-5:]  if n >= 5  else h
        b1 = sum(1 for x in h if x < 1.5)       / n * 100
        b2 = sum(1 for x in h if 1.5 <= x < 2)  / n * 100
        b3 = sum(1 for x in h if 2 <= x < 5)    / n * 100
        b4 = sum(1 for x in h if x >= 5)        / n * 100
        return {
            "total":      n,
            "avg_all":    round(statistics.mean(h), 2),
            "avg_10":     round(statistics.mean(recent10), 2),
            "avg_5":      round(statistics.mean(recent5), 2),
            "crash_rate": round(
                sum(1 for x in recent10 if x < 2.0) / len(recent10) * 100, 1
            ),
            "b1": round(b1, 1),
            "b2": round(b2, 1),
            "b3": round(b3, 1),
            "b4": round(b4, 1),
            "max_mult": round(max(h), 2),
            "min_mult": round(min(h), 2),
        }

    def get_history_list(self):
        return list(self.history)
