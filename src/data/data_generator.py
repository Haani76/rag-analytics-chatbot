import os
import json
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from configs.config import config


def generate_sales_data(num_customers: int = 50, months: int = 24) -> pd.DataFrame:
    """Generate synthetic monthly sales data per customer."""
    records = []
    customers = [f"CUST_{i:04d}" for i in range(1, num_customers + 1)]
    products = ["Product A", "Product B", "Product C", "Product D", "Product E"]
    regions = ["North", "South", "East", "West", "Central"]
    start_date = datetime(2022, 1, 1)

    for customer in customers:
        region = random.choice(regions)
        base_revenue = random.uniform(10000, 100000)
        trend = random.uniform(-0.02, 0.05)

        for month in range(months):
            date = start_date + timedelta(days=30 * month)
            seasonality = 1 + 0.2 * np.sin(2 * np.pi * month / 12)
            noise = random.uniform(0.85, 1.15)
            revenue = base_revenue * (1 + trend) ** month * seasonality * noise

            records.append({
                "date": date.strftime("%Y-%m-%d"),
                "customer_id": customer,
                "region": region,
                "product": random.choice(products),
                "revenue": round(revenue, 2),
                "units_sold": random.randint(10, 500),
                "discount_pct": round(random.uniform(0, 0.25), 2),
                "customer_segment": random.choice(["Enterprise", "SMB", "Startup"]),
            })

    return pd.DataFrame(records)


def generate_utilization_data(num_customers: int = 50, months: int = 24) -> pd.DataFrame:
    """Generate synthetic product utilization data."""
    records = []
    customers = [f"CUST_{i:04d}" for i in range(1, num_customers + 1)]
    features = ["Feature A", "Feature B", "Feature C", "Feature D"]
    start_date = datetime(2022, 1, 1)

    for customer in customers:
        base_utilization = random.uniform(0.3, 0.9)

        for month in range(months):
            date = start_date + timedelta(days=30 * month)
            utilization = min(1.0, base_utilization + random.uniform(-0.1, 0.1))

            records.append({
                "date": date.strftime("%Y-%m-%d"),
                "customer_id": customer,
                "feature": random.choice(features),
                "utilization_rate": round(utilization, 3),
                "active_users": random.randint(5, 200),
                "sessions": random.randint(50, 2000),
                "avg_session_duration_min": round(random.uniform(5, 60), 1),
            })

    return pd.DataFrame(records)


def generate_kpi_summary(sales_df: pd.DataFrame, util_df: pd.DataFrame) -> list:
    """
    Generate KPI summary documents for RAG indexing.
    These are the text chunks that will be embedded and stored.
    """
    documents = []

    # Overall sales summary
    total_revenue = sales_df["revenue"].sum()
    avg_monthly_revenue = sales_df.groupby("date")["revenue"].sum().mean()
    top_region = sales_df.groupby("region")["revenue"].sum().idxmax()
    top_product = sales_df.groupby("product")["revenue"].sum().idxmax()

    documents.append({
        "id": "overall_sales_summary",
        "title": "Overall Sales Summary",
        "content": f"""
Overall Sales Performance Summary:
Total revenue across all customers and periods: ${total_revenue:,.2f}
Average monthly revenue: ${avg_monthly_revenue:,.2f}
Top performing region: {top_region}
Top performing product: {top_product}
Total number of customers: {sales_df['customer_id'].nunique()}
Date range: {sales_df['date'].min()} to {sales_df['date'].max()}
        """.strip(),
        "category": "sales",
    })

    # Regional breakdown
    regional_revenue = sales_df.groupby("region")["revenue"].sum()
    for region, revenue in regional_revenue.items():
        pct = revenue / total_revenue * 100
        documents.append({
            "id": f"regional_summary_{region.lower()}",
            "title": f"{region} Region Sales Summary",
            "content": f"""
{region} Region Sales Summary:
Total revenue: ${revenue:,.2f}
Percentage of total revenue: {pct:.1f}%
Number of customers: {sales_df[sales_df['region']==region]['customer_id'].nunique()}
Average monthly revenue: ${sales_df[sales_df['region']==region].groupby('date')['revenue'].sum().mean():,.2f}
            """.strip(),
            "category": "sales",
        })

    # Customer segment breakdown
    segment_revenue = sales_df.groupby("customer_segment")["revenue"].sum()
    for segment, revenue in segment_revenue.items():
        pct = revenue / total_revenue * 100
        documents.append({
            "id": f"segment_summary_{segment.lower()}",
            "title": f"{segment} Segment Summary",
            "content": f"""
{segment} Customer Segment Summary:
Total revenue: ${revenue:,.2f}
Percentage of total revenue: {pct:.1f}%
Number of customers: {sales_df[sales_df['customer_segment']==segment]['customer_id'].nunique()}
Units sold: {sales_df[sales_df['customer_segment']==segment]['units_sold'].sum():,}
            """.strip(),
            "category": "sales",
        })

    # Utilization summary
    avg_utilization = util_df["utilization_rate"].mean()
    top_feature = util_df.groupby("feature")["utilization_rate"].mean().idxmax()
    documents.append({
        "id": "utilization_summary",
        "title": "Product Utilization Summary",
        "content": f"""
Product Utilization Summary:
Average utilization rate across all customers: {avg_utilization*100:.1f}%
Most used feature: {top_feature}
Average active users per customer: {util_df.groupby('customer_id')['active_users'].mean().mean():.0f}
Average sessions per month: {util_df['sessions'].mean():.0f}
Average session duration: {util_df['avg_session_duration_min'].mean():.1f} minutes
        """.strip(),
        "category": "utilization",
    })

    # Monthly trends
    monthly_revenue = sales_df.groupby("date")["revenue"].sum()
    recent_3_months = monthly_revenue.tail(3).mean()
    previous_3_months = monthly_revenue.iloc[-6:-3].mean()
    trend_pct = (recent_3_months - previous_3_months) / previous_3_months * 100

    documents.append({
        "id": "trend_summary",
        "title": "Revenue Trend Summary",
        "content": f"""
Revenue Trend Analysis:
Recent 3-month average revenue: ${recent_3_months:,.2f}
Previous 3-month average revenue: ${previous_3_months:,.2f}
Revenue trend: {trend_pct:+.1f}% change
Overall trend direction: {"Positive" if trend_pct > 0 else "Negative"}
Peak revenue month: {monthly_revenue.idxmax()}
Peak revenue: ${monthly_revenue.max():,.2f}
        """.strip(),
        "category": "trends",
    })

    return documents


def save_data(sales_df, util_df, documents):
    """Save all generated data to disk."""
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)

    sales_path = os.path.join(config.RAW_DATA_DIR, "sales_data.csv")
    util_path = os.path.join(config.RAW_DATA_DIR, "utilization_data.csv")
    docs_path = os.path.join(config.PROCESSED_DATA_DIR, "kpi_documents.json")

    sales_df.to_csv(sales_path, index=False)
    util_df.to_csv(util_path, index=False)
    with open(docs_path, "w") as f:
        json.dump(documents, f, indent=2)

    print(f"Sales data saved:       {sales_path} ({len(sales_df)} records)")
    print(f"Utilization data saved: {util_path} ({len(util_df)} records)")
    print(f"KPI documents saved:    {docs_path} ({len(documents)} documents)")

    return sales_path, util_path, docs_path


if __name__ == "__main__":
    print("Generating synthetic business data...")
    sales_df = generate_sales_data(num_customers=50, months=24)
    util_df = generate_utilization_data(num_customers=50, months=24)
    documents = generate_kpi_summary(sales_df, util_df)
    save_data(sales_df, util_df, documents)
    print("\nDone!")
    print(f"\nSample sales data:")
    print(sales_df.head(3).to_string())