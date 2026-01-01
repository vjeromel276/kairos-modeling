import alpaca_trade_api as tradeapi
api = tradeapi.REST(
    'PK347Y7OMCULH3KC5MALII6ZWP', 
    '7vceesTCANBZXXjMjEuGs1a8N1YjkAudj4aKUUXofRHB', 
    'https://paper-api.alpaca.markets'
    )
print(api.get_account().status)