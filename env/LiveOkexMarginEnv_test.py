import pytest
import mock
from live.LiveMarginTradingEnv import LiveMarginTradingEnv
from mock import patch, Mock
import pandas as pd
import json
import numpy as np
import pyarrow.parquet as pq


class TestLiveMarginTradingEnv(LiveMarginTradingEnv):
    def __init__(
        self, 
        model,
        quote_asset="btc",
        base_asset="eth",
        api_key="",
        api_secret="",
        passphrase=""
    ):
        super(TestLiveMarginTradingEnv, self).__init__(
            model,
            quote_asset,
            base_asset,
            api_key,
            api_secret,
            passphrase
        )

def test_get_account():
    pass

def test_get_config_info():
    pass

def test_borrow():
    pass

def test_repay():
    pass

def test_multiple_orders():
    pass

def _get_smallest_amount():
    pass