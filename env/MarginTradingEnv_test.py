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
    def __init__(self, df):
        super(TestMarginTradingEnv, self).__init__(df) 

    def _current_price(self):
        return 0.5

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
        (1, 2, 1, 2-(6*0.0025)),
        (1, 2, 0.5, 2-(1*0.0025)),
        # (1, 2, 0.0, ),
        # (1, 2, -0.5, ),
        # (1, 2, -1, ),
        # # all quote start
        # (2, 0, 1, ),
        # (2, 0, 0.5, ),
        # (2, 0, 0.0, ),
        # (2, 0, -0.5, ),
        # (2, 0, -1, ),
        # # all base start
        # (0, 4, 1, ),
        # (0, 4, 0.5, ),
        # (0, 4, 0.0, ),
        # (0, 4, -0.5, ),
        # (0, 4, -1, ),
        # # all base leveraged start
        # (0, 8, 1, ),
        # (0, 8, 0.5, ),
        # (0, 8, 0.0, ),
        # (0, 8, -0.5, 2-(4*0.0025)),
        # (0, 8, -1, 2-(8*0.0025)),
        # # all quote leveraged start
        # (4, 0, 1, 2-(8*0.0025)),
        # (4, 0, 0.5, ),
        # (4, 0, 0.0, ),
        # (4, 0, -0.5, ),
        # (4, 0, -1, )    
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