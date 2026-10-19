
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

data_dir = Path(__file__).resolve().parent.parent / "datasets"

df = pd.read_csv(
    data_dir / "hourly_prices.csv",
    parse_dates=["timestamp"]
)

plt.figure(figsize=(12, 5))

plt.plot(df["timestamp"], df["price"])

plt.title("AWS EC2 Spot GPU Price History")
plt.xlabel("Date")
plt.ylabel("Price (USD per GPU instance-hour)")

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(data_dir / "aws_price_history.png", dpi=300)
plt.show()
