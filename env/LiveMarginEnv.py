from lomond.persist import persist
from lomond import WebSocket
import logging
import json
import zlib

class FifoDf():
    pass

class BitcoinTradingEnv(gym.Env):
    '''A Bitcoin trading environment for OpenAI gym'''
    metadata = {'render.modes': ['human', 'system', 'none']}
    viewer = None

    def __init__(
        self, 
        agent, 
        initial_balance=10000, 
        commission=0.0025, 
        reward_func='sortino',
        close_key='close',
        date_key='timestamp_ms',
        features=['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'],
        **kwargs
    ):
        super(BitcoinTradingEnv, self).__init__()

        self.initial_balance = initial_balance
        self.commission = commission
        self.reward_func = reward_func
        self.close_key = close_key
        self.date_key = date_key

        self.forecast_len = kwargs.get('forecast_len', 10)
        self.confidence_interval = kwargs.get('confidence_interval', 0.95)
        self.obs_shape = (1, 5 + len(self.df.columns) - 2 + (self.forecast_len * 3))

        # Actions of the format Buy 1/4, Sell 3/4, Hold (amount ignored), etc.
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Observes the price action, indicators, account action, price forecasts
        self.observation_space = spaces.Box(low=0, high=1, shape=self.obs_shape, dtype=np.float16)

        self.fifo_df = ()

    def is_next():
        pass

    def _next_observation(self):
        pass

    def _take_action(self, action):
       pass

    def _reward(self):
        pass

    def _done(self):
        pass
 
    def process(self,res):
        data = res['data'][0]['candle']

        kline = {
            'timestamp': 9,
            'open': 0,
            'high': 0,
            'low': 0,
            'close': 0,
            'quote_volume': 0,
            'base_volume':0
        }

        self.fifo_df.update(kline)

        if kline['timestamp'] > self.last_time:
            action, _ = self.model.predict(self.fifo_df.get_observation())
            self.step(action)

    def reset(self):
        pass

    def step(self, action):

        self.current_account = ""
        self.current_dist = 0
        
        if action > 0:
            self.position_type = "long"



    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def run(self):
        websocket = WebSocket('wss://real.okex.com:10442/ws/v3')
        channels = ["spot/candle60s:ETH-USDT"]

        for event in persist(websocket):
            if event.name == 'poll':
                sub_param = {"op": "subscribe", "args": channels}
                sub_str = json.dumps(sub_param)
                websocket.send_text(sub_str)
            elif event.name == 'binary':
                try:
                    res = json.loads(inflate(event.data))
                    if "table" in res:
                        self.process(res)   
                    elif "error" in res:
                        logging.error(res)
                    elif "event" in res:
                        if res["event"] == "subscribe":
                            pass
                        else:
                            logging.warn(res)
                    else:
                        print(res)
                        logging.warn(res)
                except Exception as e:
                    logging.error(e)
            elif event.name == "text":
                print(event)