import os
import sys
import json
pair_list = [e for e in json.load(open('user_data\strategies\SmallCapStrategy.json'))["exchange"]["pair_whitelist"]]
exec = '/home/ftuser/.local/bin/freqtrade' if sys.platform == 'linux' else 'freqtrade'
cmd = [exec, 'download-data', 
       '-p', " ".join(pair_list),
        '-t', '1h', '1d', '15m', '4h',
        # '-t', '1m',
        # '--days', '3',
        '--timerange', '20260301-20260501',
        # '--prepend',
        "-c", 'user_data/MLBasketStrategy.json',
        '--trading-mode', 'futures']
cmd_str = ' '.join(cmd)
print(f'Running cmd: \n{cmd_str}')
os.system(cmd_str)
