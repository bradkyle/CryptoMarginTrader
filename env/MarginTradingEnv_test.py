import pytest
import mock
from env.MarginTradingEnv import MarginTradingEnv
from mock import patch, Mock
import pandas as pd
import json
import numpy as np

DF = pd.DataFrame([
{'timestamp_ms': 1511925000000,
 'open': 0.00013000,
 'high': 0.00013000,
 'low': 0.00013000,
 'close': 0.00013000,
 'base_volume': 104.00000000,
 'quote_volume': 0.01352000},
 {'timestamp_ms': 1511925000000,
 'open': 0.00013000,
 'high': 0.00013000,
 'low': 0.00013000,
 'close': 0.00013000,
 'base_volume': 104.00000000,
 'quote_volume': 0.01352000}
])

# LONG

@patch.object(MarginTradingEnv, '_current_price')
def test_all_long_no_leverage(mock_current_price):
    mock_current_price.return_value=0.5

    env = MarginTradingEnv(DF)
    env.reset()

    # 0.5 * maximum portfolio amount should
    # be placed in the long position on the
    # base asset
    # 
    env.initial_balance = 1
    env.max_leverage = 1
    env._take_action(np.array([0.5]))

    assert(env.total_value_quote==1)
    assert(env.total_value_base==2)

    assert(env.quote_held==0)
    assert(env.base_held==2)
    assert(env.base_debt==0)
    assert(env.quote_debt==0)

@patch.object(MarginTradingEnv, '_current_price')
def test_all_short_no_leverage(mock_current_price):
    mock_current_price.return_value=0.5

    env = MarginTradingEnv(DF)
    env.reset()

    # 0.5 * maximum portfolio amount should
    # be placed in the long position on the
    # base asset
    # 
    env.initial_balance = 1
    env.max_leverage = 1
    env._take_action(np.array([-0.5]))

    assert(env.total_value_quote==1)
    assert(env.total_value_base==2)

    assert(env.quote_held==1)
    assert(env.base_held==0)
    assert(env.base_debt==0)
    assert(env.quote_debt==0)

@patch.object(MarginTradingEnv, '_current_price')
def test_Long_small(mock_current_price):
    mock_current_price.return_value=0.5

    env = MarginTradingEnv(DF)
    env.reset()

    # 0.5 * maximum portfolio amount should
    # be placed in the long position on the
    # base asset
    # 
    env._take_action(np.array([0.2])) # :: 2 * 0.2 = 0.4 

    assert(env.total_value_quote==1)
    assert(env.total_value_base==2)

    assert(env.quote_held==0.8)
    assert(env.base_held==0.4)
    assert(env.base_debt==0)
    assert(env.quote_debt==0)

@patch.object(MarginTradingEnv, '_current_price')
def test_short_small(mock_current_price):
    mock_current_price.return_value=0.5

    env = MarginTradingEnv(DF)
    env.reset()

    # 0.5 * maximum portfolio amount should
    # be placed in the long position on the
    # base asset
    # 
    env._take_action(np.array([-0.2])) # :: 2 * 0.2 = 0.4 

    assert(env.total_value_quote==1)
    assert(env.total_value_base==2)

    assert(env.quote_held==0.2)
    assert(env.base_held==1.6)
    assert(env.base_debt==0)
    assert(env.quote_debt==0)

@patch.object(MarginTradingEnv, '_current_price')
def test_all_long_with_all_leverage(mock_current_price):
    mock_current_price.return_value=0.5

    env = MarginTradingEnv(DF)
    env.reset()

    # 0.5 * maximum portfolio amount should
    # be placed in the long position on the
    # base asset
    env._take_action(np.array([1]))

    assert(env.total_value_quote==1)
    assert(env.total_value_base==2)

    assert(env.quote_held==0)
    assert(env.base_held==4)
    assert(env.base_debt==0)
    assert(env.quote_debt==0)

@patch.object(MarginTradingEnv, '_current_price')
def test_all_short_with_all_leverage(mock_current_price):
    mock_current_price.return_value=0.5

    env = MarginTradingEnv(DF)
    env.reset()

    # 0.5 * maximum portfolio amount should
    # be placed in the long position on the
    # base asset
    env._take_action(np.array([-1]))

    assert(env.total_value_quote==1)
    assert(env.total_value_base==2)

    assert(env.quote_held==2)
    assert(env.base_held==0)
    assert(env.base_debt==0)
    assert(env.quote_debt==0)

@patch.object(MarginTradingEnv, '_current_price')
def test_flat_position(mock_current_price):
    mock_current_price.return_value=0.5

    env = MarginTradingEnv(DF)
    env.reset()

    # 0.5 * maximum portfolio amount should
    # be placed in the long position on the
    # base asset
    # 
    env._take_action(np.array([0.0])) # :: 2 * 0.2 = 0.4 

    assert(env.total_value_quote==1)
    assert(env.total_value_base==2)

    assert(env.quote_held==0.5)
    assert(env.base_held==1)
    assert(env.base_debt==0)
    assert(env.quote_debt==0)

# SHORT
