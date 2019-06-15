from lomond.persist import persist
from lomond import WebSocket
import logging
import json
import zlib

class FifoDf():
    pass

class LiveMarginTradingEnv(gym.Env):
    '''A Bitcoin trading environment for OpenAI gym'''
    metadata = {'render.modes': ['human', 'system', 'none']}
    viewer = None

    def __init__(
        self, 
        agent, 
        api_key,
        api_secret,
        initial_balance=10000, 
        commission=0.0025, 
        reward_func='sortino',
        base_asset='ETH',
        quote_asset='BTC',
        close_key='close',
        date_key='timestamp_ms',
        features=['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'],
        **kwargs
    ):
        super(LiveMarginTradingEnv, self).__init__()

        self.base_asset = base_asset
        self.quote_asset = quote_asset
        self.initial_balance = initial_balance
        self.commission = commission
        self.reward_func = reward_func
        self.close_key = close_key
        self.date_key = date_key
        self.instrument_id = "-".join(self.base_asset, self.quote_asset)

        self.forecast_len = kwargs.get('forecast_len', 10)
        self.confidence_interval = kwargs.get('confidence_interval', 0.95)
        self.obs_shape = (1, 5 + len(self.df.columns) - 2 + (self.forecast_len * 3))

        # Actions of the format Buy 1/4, Sell 3/4, Hold (amount ignored), etc.
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Observes the price action, indicators, account action, price forecasts
        self.observation_space = spaces.Box(low=0, high=1, shape=self.obs_shape, dtype=np.float16)

        self.fifo_df = ()

    def get_account():
        pass

    def is_next():
        pass

    def observation(self):
        pass

    def _take_action(self, action):
       pass

    def _reward(self):
        pass

    def _done(self):
        return False

    def _take_action(self, action):

        dist = action[0]
        current_price = self._current_price()

        # Get the total representative value in quote asset
        total_quote = (self.quote_held+(self.base_held*current_price))

        # Find the distribution of value in denominations 
        # representative of the quote asset
        b, q, nex_b, nex_q, lev_b, lev_q = self._balance_dist(
            dist, 
            total_quote,
            max_leverage=self.max_leverage,
            threshold=0.5
        )

        self.quote_held_wl = (self.quote_held - self.quote_debt)
        self.base_held_wl = (self.base_held - self.base_debt)

        net_worth = self.quote_held_wl + self.base_held_wl*current_price

        self.net_worths.append(net_worth)

        self.account_history = np.append(self.account_history, [
            [self.quote_held],
            [self.base_held],
            [self.base_debt],
            [self.quote_debt],
            [cost],
            [sales],
            [base_sold],
            [base_bought]
        ], axis=1)

    def _balance_dist(
        self,
        action,
        total_q,
        max_leverage=2,
        threshold=-0.5
    ):
        max_allowed = (total_q*max_leverage)

        next_q = total_q*(threshold-action)
        next_b = total_q*(threshold+action)
        
        next_q = clip(total_q, next_q)
        next_b = clip(total_q, next_b)
        
        lev_b = ((action-threshold)*2) * max_allowed
        lev_q =  -((action+threshold)*2) * max_allowed
        
        lev_b = clip(max_allowed, lev_b)
        lev_q = clip(max_allowed, lev_q)

        q = lev_q+next_q
        b = lev_b+next_b

        return (
            b,
            q,
            next_b,
            next_q,
            lev_b,
            lev_q
        )
 
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

        self.update(kline)

        if kline['timestamp'] > self.last_time:
            action, _ = self.model.predict(self.observation())
            self.step(action)

    def reset(self):
        pass

    def step(self, action):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        # Make all repayments
        # Flatten Position
        pass

    def update_account(
        self
    ):
        pass

    def update_orders(
        self
    ):
        pass

    def run(self):
        websocket = WebSocket('wss://real.okex.com:10442/ws/v3')

        channels = [
            "spot/candle60s:"+self.instrument_id,
            "spot/margin_account:"+self.instrument_id,
            "spot/order:"+self.instrument_id
        ]

        # TODO login

        for event in persist(websocket):
            if event.name == 'poll':
                sub_param = {"op": "subscribe", "args": channels}
                sub_str = json.dumps(sub_param)
                websocket.send_text(sub_str)
            elif event.name == 'binary':
                try:
                    res = json.loads(inflate(event.data))
                    if "table" in res:
                        if res["table"] == "spot/candle60s":
                            self.process(res["data"])
                        elif res["table"] == "spot/margin_account":
                            self.update_account(res["data"])
                        elif res["table"] == "spot/order":
                            self.update_orders(res["data"])   
                        elif res["table"] == "spot/trade":
                            pass  
                    elif "error" in res:
                        logging.error(res)
                    elif "event" in res:
                        if res["event"] == "subscribe":
                            pass
                        elif res["event"] == "login":
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