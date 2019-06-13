 # Base Debt Management
            if self.base_debt > next_base_debt:
                self.repayments.append({
                    'step': self.current_step,
                    'asset': 'base',
                    'quantity': self.base_debt-next_base_debt
                })
            else:
                self.loans.append({
                    'step': self.current_step,
                    'asset': 'base',
                    'quantity': next_base_debt-self.quote_debt
                })

            # Quote Debt Management
            if self.quote_debt > next_quote_debt:
                self.repayments.append({
                    'step': self.current_step,
                    'asset': 'quote',
                    'quantity': self.quote_debt-next_quote_debt
                })
            else:
                self.loans.append({
                    'step': self.current_step,
                    'asset': 'quote',
                    'quantity': next_quote_debt-self.quote_debt
                })

            if self.base_held > next_base_held:
                self.trades.append({
                    'step': self.current_step,
                    'quantity': self.base_held - next_base_held, 
                    'price': current_price,
                    'type': 'sell'
                })
            else:
                self.trades.append({
                    'step': self.current_step,
                    'quantity': next_base_held - self.base_held, 
                    'price': current_price,
                    'type': 'buy'
                })

            if self.quote_held > next_quote_held:
                self.trades.append({
                    'step': self.current_step,
                    'quantity': self.quote_held - next_quote_held, 
                    'price': current_price,
                    'type': 'sell'
                })
            else:
                self.trades.append({
                    'step': self.current_step,
                    'quantity': next_quote_held - self.next_quote_held, 
                    'price': current_price,
                    'type': 'buy'
                })