import pytest
import mock
from env.MarginTradingEnv import MarginTradingEnv
from mock import patch, Mock
import pandas as pd
import json
import numpy as np
import pyarrow.parquet as pq

DF = pq.read_table('./data/test.parquet').to_pandas()
DF.sort_values(['window_end'], inplace=True)

class TestMarginTradingEnv(MarginTradingEnv):
    def __init__(self, df, current_price=0.5):
        super(TestMarginTradingEnv, self).__init__(df)
        self.current_price = current_price

    def _current_price(self):
        return self.current_price

@pytest.mark.parametrize(
    "quote_held,base_held,action,expected_base,expected_quote", 
    [
    # equal start
    (1, 2, 1, 8, 0),
    (1, 2, 0.5, 4, 0),
    (1, 2, 0.0, 2, 1),
    (1, 2, -0.5, 0, 2),
    (1, 2, -1, 0, 4),
    # all quote start
    (2, 0, 1, 8, 0),
    (2, 0, 0.5, 4, 0),
    (2, 0, 0.0, 2, 1),
    (2, 0, -0.5, 0, 2),
    (2, 0, -1, 0, 4),
    # all base start
    (0, 4, 1, 8, 0),
    (0, 4, 0.5, 4, 0),
    (0, 4, 0.0, 2, 1),
    (0, 4, -0.5, 0, 2),
    (0, 4, -1, 0, 4),
    # all base leveraged start
    (0, 8, 1, 8, 0),
    (0, 8, 0.5, 4, 0),
    (0, 8, 0.0, 2, 1),
    (0, 8, -0.5, 0, 2),
    (0, 8, -1, 0, 4),
    # all quote leveraged start
    (4, 0, 1, 8, 0),
    (4, 0, 0.5, 4, 0),
    (4, 0, 0.0, 2, 1),
    (4, 0, -0.5, 0, 2),
    (4, 0, -1, 0, 4),
])
def test_take_action_without_commission(
    quote_held, 
    base_held, 
    action,
    expected_base,
    expected_quote
):
   
    env = TestMarginTradingEnv(DF)
    env.reset()
    env.commission = 0.0
    env.max_leverage = 1

    env.initial_balance = quote_held + base_held
    env.quote_held = quote_held
    env.base_held = base_held
    env.base_debt = max(base_held-4, 0)
    env.quote_debt = max(quote_held-2, 0)
    env._take_action(np.array([action]))

    assert(env.quote_held==expected_quote)
    assert(env.base_held==expected_base)


@pytest.mark.parametrize(
    "quote_held,base_held,action,expected_total_value", 
    [
        # equal start
        (1, 2, 1, 2-(3*0.0025)),
        (1, 2, 0.5, 2-(1*0.0025)),
        (1, 2, 0.0, 2),
        (1, 2, -0.5, 2-(1*0.0025)),
        (1, 2, -1, 2-(3*0.0025)),
        # # all quote start
        (2, 0, 1, 2-(4*0.0025)),
        (2, 0, 0.5, 2-(2*0.0025)),
        (2, 0, 0.0, 2-(1*0.0025)),
        (2, 0, -0.5, 2),
        (2, 0, -1, 2-(2*0.0025)),
        # # all base start
        (0, 4, 1, 2-(2*0.0025)),
        (0, 4, 0.5, 2),
        (0, 4, 0.0, 2-(1*0.0025)),
        (0, 4, -0.5, 2-(2*0.0025)),
        (0, 4, -1, 2-(4*0.0025)),
        # # all base leveraged start
        (0, 8, 1, 2),
        (0, 8, 0.5, 2-(2*0.0025)),
        (0, 8, 0.0, 2-(3*0.0025)),
        (0, 8, -0.5, 2-(4*0.0025)),
        (0, 8, -1, 2-(4*0.0025)),
        # # all quote leveraged start
        (4, 0, 1, 2-(4*0.0025)),
        (4, 0, 0.5, 2-(4*0.0025)), # 
        (4, 0, 0.0, 2-(3*0.0025)), # -> 
        (4, 0, -0.5, 2-(2*0.0025)),
        (4, 0, -1, 2)    
])
def test_take_action_with_commission(
    quote_held, 
    base_held, 
    action,
    expected_total_value
):
   
    env = TestMarginTradingEnv(DF)
    env.reset()
    env.commission = 0.0025
    env.max_leverage = 1

    env.initial_balance = quote_held + base_held
    env.quote_held = quote_held
    env.base_held = base_held
    env.base_debt = max(base_held-4, 0)
    env.quote_debt = max(quote_held-2, 0)
    env._take_action(np.array([action]))

    assert(env.total_value_minus_debt==expected_total_value)


@pytest.mark.parametrize(
    "quote_held,base_held,price_1,price_2,expected_value", 
    [
        (1, 2, 0.5, 0.25, 1.5),
        (2, 0, 0.5, 0.25, 2),
        (0, 4, 0.5, 0.25, 1),
        (0, 8, 0.5, 0.25, 0), #
        (4, 0, 0.5, 0.25, 3), # short 
        (1, 2, 0.5, 1, 3), 
        (2, 0, 0.5, 1, 2),
        (0, 4, 0.5, 1, 4), 
        (0, 8, 0.5, 1, 6), #
        (4, 0, 0.5, 1, 0),
    ]
)
def test_value_change(
    quote_held,
    base_held, 
    price_1,
    price_2,
    expected_value
):
    env = TestMarginTradingEnv(DF)
    env.current_price = price_1
    env.reset()
    env.commission = 0.0025
    env.max_leverage = 1

    tot = ((base_held*price_1)-quote_held)/4
    print(tot)

    env.initial_balance = quote_held + base_held
    env.quote_held = quote_held
    env.base_held = base_held
    env.base_debt = max(base_held-(2/price_1), 0)
    env.quote_debt = max(quote_held-2, 0)
    env.get_value()
    env._take_action(np.array([tot]))
    
    env.current_price = price_2
    _, tvmd, _ = env.get_value()

    assert(tvmd==expected_value)
    # assert(1==2)


@pytest.mark.parametrize(
    "quote_held,base_held,price_1,price_2,price_3", 
    [
        (1, 2, 0.5, 0.1, 0.5)
           
    ]
)
def test_drastic_swing_in_price(
    quote_held,
    base_held, 
    price_1,
    price_2,
    price_3
):
    env = TestMarginTradingEnv(DF)
    env.reset()
    env.commission = 0.0025
    env.max_leverage = 1

    env.initial_balance = quote_held + base_held
    env.quote_held = quote_held
    env.base_held = base_held
    env.base_debt = max(base_held-4, 0)
    env.quote_debt = max(quote_held-2, 0)
    env._take_action(np.array([0.0]))

    assert(env.total_value_minus_debt==quote_held+base_held*0.5)

# @pytest.mark.parametrize(
#     "quote_held,base_held,price_1,price_2,price_3,action", 
#     [
#         ()
           
#     ]
# )
# def test_exiting_leverage():
#    pass

# @pytest.mark.parametrize(
#     "quote_held,base_held,price_1,price_2,price_3,action", 
#     [
#         ()
           
#     ]
# )
# def test_drastic_swing_in_price_with_action():
#    pass

@pytest.mark.parametrize(
    "quote_held,base_held,price_1,price_2,price_3", 
    [
        (1, 2, 0.5, 0.1, 0.5)
           
    ]
)
def test_drastic_swing_in_price(
    quote_held,
    base_held, 
    price_1,
    price_2,
    price_3
):
    env = TestMarginTradingEnv(DF)
    env.reset()
    env.commission = 0.0025
    env.max_leverage = 1

    env.initial_balance = quote_held + base_held
    env.quote_held = quote_held
    env.base_held = base_held
    env.base_debt = max(base_held-4, 0)
    env.quote_debt = max(quote_held-2, 0)
    env._take_action(np.array([0.0]))

    assert(env.total_value_minus_debt==quote_held+base_held*0.5)
