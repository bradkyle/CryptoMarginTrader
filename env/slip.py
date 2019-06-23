
b=b_dist/current_price
        q=q_dist
        sales = 0
        cost = 0
        base_sold = 0
        base_bought = 0

        # TODO simulate loss 
        # TODO simulate gain

        b_delta = self.base_held - b_dist
        q_delta = self.quote_held - q_dist

        if b_delta > q_delta:
            print("bazinga")
        
        elif q_delta > b_delta:
            print("slappy")

        elif q_delta == b_delta:
            print("sloppy")

        if self.base_held > b and self.quote_held > 0:
            base_sold = self.base_held - b

            self.trades.append({
                'step': self.current_step,
                'quantity': base_sold, 
                'price': current_price,
                'type': 'sell'
            })

            price = current_price * (1 - self.commission)
            sales =  base_sold * price 

            self.base_held -= base_sold
            self.quote_held += sales
            
        elif self.base_held < b and self.quote_held > 0:
            base_bought = b - self.base_held

            self.trades.append({
                'step': self.current_step,
                'quantity': base_bought, 
                'price': current_price,
                'type': 'buy'
            })

            price = current_price * (1 + self.commission)
            cost =  ((base_bought * price) - lev_q)

            self.base_held += base_bought
            self.quote_held -= cost

        elif self.quote_held > q:
            print("slippage")
        elif self.quote_held < q:
            print("sprinksle")

        self.base_debt = lev_b
        self.quote_debt = lev_q
        self.total_debt = (self.quote_debt+self.base_debt*current_price)
        self.total_value = (self.quote_held + self.base_held*current_price)
        self.total_value_minus_debt = self.total_value - self.total_debt

        self.net_worths.append(self.total_value_minus_debt)

       print("="*80)
        print("step: "+str(self.current_step))
        print("price change: "+str(abs(round((current_price-self.prev_price)/self.prev_price, 6))))
        print("net worth: "+str(self.total_value_minus_debt))
        print("side: "+ self.side)
        print("current_price: "+str(current_price))
        print("price: "+str(price))
        print("action: "+str(action))
        print("quote held: "+str(self.quote_held))
        print("base held: "+str(self.base_held))
        print("lev_b_delta: "+str(lev_b_delta))
        print("lev_q_delta: "+str(lev_q_delta))
        print("total debt: "+str(self.total_debt))
        print("quote debt: "+str(self.quote_debt))
        print("base debt: "+str(self.base_debt))
        print("base sold: "+str(base_sold))
        print("base bought: "+str(base_bought))
        print("cost: "+str(cost))
        print("sales: "+str(sales))
        print("q_delta: "+ str(q_delta))
        print("b_delta: "+ str(b_delta))
        print("t_delta: "+ str(t_delta))
        print("nex_b: "+ str(nex_b))
        print("nex_q: "+ str(nex_q))
        print("lev_b: "+ str(lev_b))
        print("lev_q: "+ str(lev_q))
        print("b: "+ str(b))
        print("q: "+ str(q))
        print("done: "+str(self._done()))
        print("reward: " +(str(self._reward())))
        print("="*80)

        if self.base_debt > lev_b:
                self.repayments.append({
                    'step': self.current_step,
                    'asset': 'base',
                    'quantity': self.base_debt-lev_b
                })
        else:
            self.loans.append({
                'step': self.current_step,
                'asset': 'base',
                'quantity': lev_b-self.base_debt
            })

        # Quote Debt Management
        if self.quote_debt > lev_q:
            self.repayments.append({
                'step': self.current_step,
                'asset': 'quote',
                'quantity': self.quote_debt-lev_q
            })
        else:
            self.loans.append({
                'step': self.current_step,
                'asset': 'quote',
                'quantity': lev_q-self.quote_debt
            })

        if self.base_held > b:
            self.trades.append({
                'step': self.current_step,
                'quantity': self.base_held - b, 
                'price': current_price,
                'type': 'sell'
            })
        else:
            self.trades.append({
                'step': self.current_step,
                'quantity': b - self.base_held, 
                'price': current_price,
                'type': 'buy'
            })

def run(self):
    
        channels = [
            "spot/candle60s:"+self.instrument_id,
            "spot/margin_account:"+self.instrument_id
        ]

        for event in persist(self.websocket):
            if event.name == 'poll':

                timestamp = str(self.server_timestamp())
                login_str = self.login_params(
                    str(timestamp), 
                    self.api_key, 
                    self.passphrase, 
                    self.secret_key
                )

                self.websocket.send(login_str)

                sub_param = {"op": "subscribe", "args": channels}
                sub_str = json.dumps(sub_param)
                self.websocket.send_text(sub_str)

            elif event.name == 'binary':
                try:
                    res = json.loads(self.inflate(event.data))
                    if "table" in res:
                        if res["table"] == "spot/margin_account":
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

    def login_params(self, timestamp, api_key, passphrase, secret_key):
        message = timestamp + 'GET' + '/users/self/verify'
        mac = hmac.new(bytes(secret_key, encoding='utf8'), bytes(message, encoding='utf-8'), digestmod='sha256')
        d = mac.digest()
        sign = base64.b64encode(d)

        login_param = {"op": "login", "args": [api_key, passphrase, timestamp, sign.decode("utf-8")]}
        login_str = json.dumps(login_param)
        return login_str

    def get_server_time(self):
        url = "http://www.okex.com/api/general/v3/time"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()['iso']
        else:
            return ""

    def server_timestamp(self):
        server_time = self.get_server_time()
        parsed_t = dp.parse(server_time)
        timestamp = parsed_t.timestamp()
        return timestamp

    def inflate(self, data):
        decompress = zlib.decompressobj(
                -zlib.MAX_WBITS  # see above
        )
        inflated = decompress.decompress(data)
        inflated += decompress.flush()
        return inflated
