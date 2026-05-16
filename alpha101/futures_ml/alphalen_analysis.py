import alphalens
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .data import build_full_dataset
from . import config
if __name__ == "__main__":
    cfg = config.get_config()
    df = build_full_dataset(
        cfg.pairs,
        cfg.lookback_days,
        cfg.data_root,
        cfg.timeframe,
        test_start_date=cfg.test_start_date,
        test_end_date=cfg.test_end_date,
    )
    alpha_cols = [col for col in df.columns if col.startswith("alpha")]
    prices = df[['date','symbol','close']].pivot(index='date', columns='symbol', values='close')
    prices.index.freq = pd.infer_freq(prices.index)
    # Ingest and format data
    for i in range(len(alpha_cols)):
        print(f"### Analyzing {alpha_cols[i]}...")
        try:
            factor = df.set_index(['date', 'symbol'])[alpha_cols[i]]
            factor.index.levels[0].freq = pd.infer_freq(prices.index)
            factor_data = alphalens.utils.get_clean_factor_and_forward_returns(factor,
                                                                            prices,
                                                                            quantiles=5)

            # Run analysis
            alphalens.tears.create_full_tear_sheet(factor_data)
        except Exception as e:
            print(f"Error analyzing {alpha_cols[i]}: {e}")
        print("\n\n")
        continue
