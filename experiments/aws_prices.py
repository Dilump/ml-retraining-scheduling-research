
import boto3
import pandas as pd
from datetime import datetime, timedelta, timezone

ec2 = boto3.client("ec2", region_name="us-east-1")

end = datetime.now(timezone.utc)
start = end - timedelta(days=30)

paginator = ec2.get_paginator("describe_spot_price_history")

rows = []

for page in paginator.paginate(
    InstanceTypes=["g4dn.xlarge"],
    ProductDescriptions=["Linux/UNIX"],
    StartTime=start,
    EndTime=end
):
    for p in page["SpotPriceHistory"]:
        rows.append({
            "timestamp": p["Timestamp"],
            "price": float(p["SpotPrice"]),
            "instance_type": p["InstanceType"],
            "availability_zone": p["AvailabilityZone"]
        })

df = pd.DataFrame(rows)

if df.empty:
    raise RuntimeError("No price data returned.")

df = df.sort_values("timestamp")
df.to_csv("datasets/aws_spot_prices.csv", index=False)

print(df.head())
print("Records collected:", len(df))
