# MSFT Stock Price Forecasting

Predicts the next-day closing price of Microsoft (MSFT) using the last
5 days of closing prices from 20 correlated stocks (100 input features).

## Models

| Model | Params | Test MSE |
|---|---|---|
| LSTM (Teacher) | 217,217 | 11,142 |
| Attention (Transformer) | 70,401 | 443 |
| LSTM Student (Knowledge Distillation) | 24,129 | 5,595 |
| LSTM Student (No KD baseline) | 24,129 | 1,332 |
| Mamba (State Space Model) | 67,585 | 659 |
| **Extra Credit: Normalized Transformer** | **53,569** | **< 20** |

## Key Idea (Extra Credit)
Per-stock normalization ("RevIN-lite") transforms raw prices into
relative daily ratios before feeding them to the model. The model
learns price movements (~±1–2%) rather than absolute levels,
dramatically stabilizing training and reducing MSE.

## Stack
Python · PyTorch · PyTorch Lightning · yfinance
