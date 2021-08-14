import matplotlib.pyplot as plt
import mpl_finance
import numpy as np
from matplotlib.gridspec import GridSpec
import talib
import ccxt
import time

binance_futures = ccxt.binance({ 'options': { 'defaultType': 'future' } })
ticker = binance_futures.fetch_ticker('BTC/USDT')

#ohlcv
#[[time, open, high, low, close, volume], [], []]
#   0      1     2    3     4       5
def first_5candle():
	candles = binance_futures.fetch_ohlcv('BTC/USDT', timeframe='1m', limit=7)
	return candles

#그래프 그리는 함수를 하나로 합치기: ok
def ccxt_graph():
    #custom_open,custom_high,custom_low,custom_close,custom_volume,custom_date,pd_close_5m = custom_resmapler(
    #    data_1m, start-abs((finish-start)-candles_5min)+1, finish, timedelta_x_min)
    #if(len(custom_open)==0):
    #    return
	
    ohlcv_1m = binance_futures.fetch_ohlcv('BTC/USDT', timeframe='1m', limit=15)
    ohlcv_1m_for_ma = binance_futures.fetch_ohlcv('BTC/USDT',timeframe='1m', limit=115)
    ohlcv_5m = binance_futures.fetch_ohlcv('BTC/USDT', timeframe='5m', limit=10)
    ohlcv_5m_for_ma = binance_futures.fetch_ohlcv('BTC/USDT',timeframe='5m', limit=110)

    open_1m,high_1m,low_1m,close_1m,volume_1m,date_1m,close_1m_pd = [],[],[],[],[],[],[]
    open_5m,high_5m,low_5m,close_5m,volume_5m,date_5m,close_5m_pd = [],[],[],[],[],[],[]
    for x in range(15):
        open_1m.append(float(ohlcv_1m[x][1]))
        high_1m.append(float(ohlcv_1m[x][2]))
        low_1m.append(float(ohlcv_1m[x][3]))
        close_1m.append(float(ohlcv_1m[x][4]))
        volume_1m.append(float(ohlcv_1m[x][5]))
        date_1m.append(ohlcv_1m[x][0])
    for x in range(115):
        close_1m_pd.append(float(ohlcv_1m_for_ma[x][4]))
    for x in range(10):
        open_5m.append(float(ohlcv_5m[x][1]))
        high_5m.append(float(ohlcv_5m[x][2]))
        low_5m.append(float(ohlcv_5m[x][3]))
        close_5m.append(float(ohlcv_5m[x][4]))
        volume_5m.append(float(ohlcv_5m[x][5]))
        date_5m.append(ohlcv_5m[x][0])
    for x in range(110):
        close_5m_pd.append(float(ohlcv_5m_for_ma[x][4]))

    MA_7_1m = talib.MA(np.array(close_1m_pd), timeperiod=7)
    MA_25_1m = talib.MA(np.array(close_1m_pd), timeperiod=25)
    MA_99_1m = talib.MA(np.array(close_1m_pd), timeperiod=99)
    MA_7_5m = talib.MA(np.array(close_5m_pd), timeperiod=7)
    MA_25_5m = talib.MA(np.array(close_5m_pd), timeperiod=25)
    MA_99_5m = talib.MA(np.array(close_5m_pd), timeperiod=99)

    fig = plt.figure(num=1, figsize=(5, 5), dpi=100, facecolor='w', edgecolor='k') # figsize: ppi dpi: 해상도
    gs = GridSpec(nrows=12, ncols=1)######비율
    dx = fig.add_subplot(gs[0:5, 0]) #111은 subplot 그리드 인자를 정수 하나에 다 모아서 표현한 것.(1x1그리드에 첫 번째 subplot)
    ax = fig.add_subplot(gs[5, 0]) #볼륨차트 추가
    bx = fig.add_subplot(gs[6:11, 0])
    cx = fig.add_subplot(gs[11, 0])
    mpl_finance.volume_overlay(ax, open_5m, close_5m, volume_5m, width=0.4, colorup='r', colordown='b', alpha=1)
    mpl_finance.candlestick2_ochl(dx, open_5m,close_5m,high_5m,low_5m, width=0.965, colorup='r', colordown='b', alpha=1)
    mpl_finance.volume_overlay(cx, open_1m, close_1m, volume_1m, width=0.4, colorup='r', colordown='b', alpha=1)
    mpl_finance.candlestick2_ochl(bx, open_1m, close_1m, high_1m, low_1m, width=0.965, colorup='r', colordown='b', alpha=1)
    plt.autoscale() #자동 스케일링
    dx.plot(MA_7_5m[99:], color='gold', linewidth=1, alpha=1)#99+~이니까 99부터 시작
    dx.plot(MA_25_5m[99:], color='violet', linewidth=1, alpha=1)
    dx.plot(MA_99_5m[99:], color='green', linewidth=1, alpha=1)
    bx.plot(MA_7_1m[99:], color='gold', linewidth=1.5, alpha=1)
    bx.plot(MA_25_1m[99:], color='violet', linewidth=1.5, alpha=1)
    bx.plot(MA_99_1m[99:], color='green', linewidth=1.5, alpha=1)
    plt.axis('off')  # 상하좌우 축과 라벨 모두 제거
    ax.axis('off')
    dx.axis('off')
    bx.axis('off')
    cx.axis('off')
    plt.savefig('../test_data/ccxt_binance_test.jpg', bbox_inches='tight')#uuid.uuid4()
    #plt.show()
    plt.cla() #좌표축을 지운다.
    plt.clf() #현재 Figure를 지운다.

if __name__ == "__main__":
    while(1):
        ccxt_graph()
        time.sleep(1)