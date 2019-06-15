
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