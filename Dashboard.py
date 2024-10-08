import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Cars Sales Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


def visualization():
    import numpy as np
    import pandas as pd
    import streamlit as st
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.graph_objects as go
    import plotly.express as px
    import joblib
    from xgboost import XGBRegressor
    from sklearn.preprocessing import LabelEncoder
    import streamlit as st
    from streamlit_option_menu import option_menu
    import os

    df = pd.read_csv("Cars sales.csv")
    st.sidebar.header("Cars Sales Dashboard")
    st.sidebar.image("Car.jpg")
    st.sidebar.write(
        "This is a simple dashboard to analyze the car sales data :two_hearts:"
    )
    st.markdown(
        """
        <style>
        body {
            background-color: Black;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <style>
            .metric-box {
                border: 2px solid #00468B;
                border-radius: 8px;
                padding: 20px;
                text-align: center;
            }
            .metric-label {
                font-size: 1.2em;
                font-weight: bold;
            }
            .metric-value {
                font-size: 2em;
            }
        </style>
    """,
        unsafe_allow_html=True,
    )
    metrics = {
        "Total Price": df["Price"].sum(),
        "Total Tax": df["Tax"].sum(),
        "Total Mileage": df["Mileage"].sum(),
        "Count Models": df["Model"].nunique(),
    }

    def abbreviate_number(num):
        suffixes = {1_000_000_000: "B", 1_000_000: "M", 1_000: "K"}
        for key, suffix in suffixes.items():
            if abs(num) >= key:
                return f"{num / key:.2f}{suffix}"
        return str(num)

    A1, A2, A3, A4 = st.columns(4)
    columns = [A1, A2, A3, A4]
    for col, (label, value) in zip(columns, metrics.items()):
        formatted_value = abbreviate_number(value)
        with col:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{formatted_value}</div>
                </div>
            """,
                unsafe_allow_html=True,
            )
    st.write("______________")
    selected_level = st.sidebar.selectbox(
        "Model", ["None"] + list(df["Model"].unique())
    )
    selected_paid = st.sidebar.selectbox("Brand", ["None"] + list(df["Brand"].unique()))
    selected_subject = st.sidebar.selectbox(
        "Transmission", ["None"] + list(df["Transmission"].unique())
    )
    selected_year = st.sidebar.selectbox(
        "Fuel Type", ["None"] + list(df["Fuel Type"].unique())
    )
    filtered_df = df
    if selected_level != "None":
        filtered_df = filtered_df[filtered_df["Model"] == selected_level]
    if selected_paid != "None":
        filtered_df = filtered_df[filtered_df["Brand"] == selected_paid]
    if selected_subject != "None":
        filtered_df = filtered_df[filtered_df["Transmission"] == selected_subject]
    if selected_year != "None":
        filtered_df = filtered_df[filtered_df["Fuel Type"] == selected_year]
    st.write("## Cars Sales Data")
    st.write(filtered_df)
    st.write("______________")
    st.write("### Statistical transactions with data")
    st.write(df.describe().T)
    st.write(df.describe(include="object").T)
    st.write("______________")
    st.write("## Sales Dashboard 📊")
    grouped_data = df.groupby("Year")["Price"].sum().reset_index()
    fig = px.line(
        grouped_data,
        x="Year",
        y="Price",
        title="Total Price Over Years",
        labels={"year": "Year", "price": "Total Price"},
        line_shape="spline",
        markers=True,
    )
    fig.update_layout(
        title_font_size=18,
        xaxis_title_font_size=18,
        yaxis_title_font_size=18,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_traces(
        hovertemplate="<b>Year: %{x}</b><br>Total Price: %{y:,.2f}<extra></extra>",
        line=dict(color="Orange", width=3),
    )
    st.plotly_chart(fig, use_container_width=True)
    total_price_by_brand = df.groupby("Brand")["Price"].sum().reset_index()
    total_price_by_brand = total_price_by_brand.sort_values(by="Price", ascending=False)
    fig = px.bar(
        total_price_by_brand,
        x="Brand",
        y="Price",
        title="Total Price Over Brand",
        labels={"Brand": "Brand", "Price": "Total Price"},
        text="Price",
        height=500,
        color="Price",
        color_continuous_scale=px.colors.sequential.Oranges,
    )
    fig.update_traces(texttemplate="%{text:.2s}", textposition="outside")
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide")
    fig.update_layout(
        title_font_size=18,
        xaxis_title_font_size=18,
        yaxis_title_font_size=18,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_traces(
        hovertemplate="<b>Brand: %{x}</b><br>Total Price: %{y:,.2f}<extra></extra>"
    )
    st.plotly_chart(fig, use_container_width=True)
    d1, d2, d3 = st.columns([2, 0.3, 2])
    with d1:
        fuel_data = df.groupby("Fuel Type")["Price"].sum().reset_index()
        fuel_data = fuel_data.sort_values(by="Price")
        labels = fuel_data["Fuel Type"]
        values = fuel_data["Price"]
        colors = px.colors.sequential.Darkmint[-len(labels) :]
        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.4,
                    textinfo="label+percent",
                    marker=dict(colors=colors),
                    hoverinfo="label+value",
                )
            ]
        )
        fig_pie.update_layout(
            title="Total Price Over Fuel Type",
            title_font=dict(size=18),
            annotations=[
                dict(text="Fuel Type", x=0.5, y=0.5, font_size=20, showarrow=False)
            ],
            showlegend=True,
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    with d3:
        transmission_data = df.groupby("Transmission")["Price"].sum().reset_index()
        transmission_data = transmission_data.sort_values(by="Price", ascending=False)
        color_scale = px.colors.sequential.Oranges
        transmission_data["color_index"] = (
            transmission_data["Price"] / transmission_data["Price"].max()
        )
        transmission_data["color"] = transmission_data["color_index"].apply(
            lambda x: color_scale[int(x * (len(color_scale) - 1))]
        )
        fig_histogram = go.Figure(
            go.Bar(
                x=transmission_data["Transmission"],
                y=transmission_data["Price"],
                marker=dict(color=transmission_data["color"]),
                text=[f"${price:.2f}" for price in transmission_data["Price"]],
                textposition="outside",
            )
        )
        fig_histogram.update_layout(
            title="Total Price by Transmission",
            xaxis_title="Transmission Type",
            yaxis_title="Total Price",
            title_font_size=18,
            xaxis_title_font_size=18,
            yaxis_title_font_size=18,
            margin=dict(l=40, r=40, t=60, b=40),
            bargap=0.2,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_histogram, use_container_width=True)
    v1, v2, v3 = st.columns([2, 0.3, 2])
    with v1:
        mileage_mean = df["Mileage"].mean()
        fig = px.histogram(
            df,
            x="Mileage",
            y="Transmission",
            nbins=30,
            title="Mileage over Transmission",
            color_discrete_sequence=["indianred"],
            labels={"Mileage": "Mileage", "Transmission": "Transmission"},
        )
        fig.update_layout(xaxis_title="Mileage", yaxis_title="Transmission", bargap=0.2)
        fig.add_vline(x=mileage_mean, line_dash="dash", line_color="black")
        st.plotly_chart(fig, use_container_width=True)
    with v3:
        avg_price_fuel = df.groupby("Fuel Type")["Price"].mean().reset_index()
        avg_price_fuel = avg_price_fuel.sort_values(by="Price", ascending=False)
        fig = px.bar(
            avg_price_fuel,
            x="Fuel Type",
            y="Price",
            title="Average Price by Fuel Type",
            color_discrete_sequence=["orange"],
        )
        fig.update_layout(
            title_font_size=20,
            xaxis_title="Fuel Type",
            yaxis_title="Average Price",
            font=dict(size=12, color="black"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            bargap=0.2,
        )
        st.plotly_chart(fig, use_container_width=True)
    fig = px.box(
        df,
        x="Fuel Type",
        y="Mileage",
        title="Mileage Over Fuel Type",
        color_discrete_sequence=["orange"],
    )
    fig.update_layout(
        title_font_size=20,
        xaxis_title="Fuel Type",
        yaxis_title="Mileage",
        font=dict(size=12, color="black"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
    fig = px.box(
        df,
        x="Fuel Type",
        y="Engine Size",
        title="Engine Size Over Fuel Type",
        color_discrete_sequence=["white"],
    )
    fig.update_layout(
        title_font_size=20,
        xaxis_title="Fuel Type",
        yaxis_title="Engine Size",
        font=dict(size=12, color="black"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
    fig = px.box(
        df,
        x="Transmission",
        y="Engine Size",
        title="Engine Size Over Transmission",
        color_discrete_sequence=["orange"],
    )
    fig.update_layout(
        title_font_size=20,
        xaxis_title="Transmission",
        yaxis_title="Engine Size",
        font=dict(size=12, color="black"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
    fig = px.box(
        df,
        x="Brand",
        y="Mileage",
        title="Mileage Over Brand",
        color_discrete_sequence=["magenta"],
    )
    fig.update_layout(
        title_font_size=20,
        xaxis_title="Brand",
        yaxis_title="Mileage",
        font=dict(size=12, color="black"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
    avg_values_year = df.groupby("Year")[["MPG", "Engine Size"]].mean().reset_index()
    fig = px.line(
        avg_values_year,
        x="Year",
        y="MPG",
        title="Average MPG and Engine Size by Year",
        line_shape="linear",
    )
    fig.update_traces(
        x=avg_values_year["Year"],
        y=avg_values_year["MPG"],
        mode="lines+markers",
        name="MPG",
        line=dict(color="Orange", width=3),
        marker=dict(size=8),
        yaxis="y1",
    )
    fig.add_scatter(
        x=avg_values_year["Year"],
        y=avg_values_year["Engine Size"],
        mode="lines+markers",
        name="Engine Size",
        line=dict(color="red", width=3),
        marker=dict(size=8),
        yaxis="y2",
    )
    fig.update_layout(
        title_font_size=20,
        xaxis_title="Year",
        yaxis=dict(
            title="MPG",
            titlefont=dict(color="Orange"),
            tickfont=dict(color="Orange"),
            overlaying="y",
            side="left",
        ),
        yaxis2=dict(
            title="Engine Size",
            titlefont=dict(color="red"),
            tickfont=dict(color="red"),
            overlaying="y",
            side="right",
        ),
        font=dict(size=12, color="black"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
    fig = px.scatter(
        df,
        x="MPG",
        y="Price",
        size="Engine Size",
        color="Brand",
        title="Price Over MPG",
        color_continuous_scale="oranges",
        hover_name="Brand",
        size_max=60,
        labels={"MPG": "Miles Per Gallon", "Price": "Price ($)"},
    )
    fig.update_layout(
        title_font_size=20,
        xaxis_title="Miles Per Gallon (MPG)",
        yaxis_title="Price ($)",
        font=dict(size=12, color="black"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
    fig = px.scatter(
        df,
        x="Mileage",
        y="Price",
        size="Tax",
        color="Brand",
        title="Price Over Mileage",
        color_continuous_scale="oranges",
        hover_name="Brand",
        size_max=60,
        labels={"Mileage": "Mileage (km)", "Price": "Price ($)"},
    )
    fig.update_layout(
        title_font_size=20,
        xaxis_title="Mileage (km)",
        yaxis_title="Price ($)",
        font=dict(size=12, color="black"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
    f1, f2, f3 = st.columns([2, 0.3, 2])
    with f1:
        fig_fuel = px.pie(
            df,
            names="Fuel Type",
            title="Fuel Type Distribution",
            color_discrete_sequence=px.colors.sequential.Blues[::-1],
        )
        fig_fuel.update_traces(
            textinfo="percent+label",
            marker=dict(line=dict(color="black", width=2)),
        )
        fig_fuel.update_layout(
            title_font_size=24,
            legend_title_text="Fuel Type",
            legend=dict(x=0.8, y=0.5),
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_fuel, use_container_width=True)
    with f3:
        fig_transmission = px.pie(
            df,
            names="Transmission",
            title="Transmission Types Distribution",
            color_discrete_sequence=px.colors.sequential.Oranges[::-1],
        )
        fig_transmission.update_traces(
            textinfo="percent+label",
            marker=dict(line=dict(color="black", width=2)),
        )
        fig_transmission.update_layout(
            title_font_size=24,
            legend_title_text="Transmission Types",
            legend=dict(x=0.8, y=0.5),
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_transmission, use_container_width=True)
    st.write("______________")
    st.write("## About")
    st.write(
        "#### This dashboard was created by [Eng. Mohamed Nasr](https://www.linkedin.com/in/medoox369)."
    )
    st.write(
        "#### The code for this dashboard is available on [GitHub](https://github.com/medoox369/Cars_sales/)"
    )
    st.write("______________")
    st.write("## End of Dashboard")
    st.write("______________")
    st.write("## Thank You :smile:")
    st.write("______________")


def report():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.graph_objects as go
    import plotly.express as px
    import joblib
    from xgboost import XGBRegressor
    from sklearn.preprocessing import LabelEncoder
    import streamlit as st
    from streamlit_option_menu import option_menu
    import os
    import streamlit as st
    import joblib
    import os

    st.write("# Report")
    st.write("______________")
    st.write("## Introduction")
    st.write(
        """This article presents a thorough analysis of car sales data, uncovering key insights and offering practical recommendations to guide future business strategies. With the continuous evolution of the automotive market, it is essential to delve into trends, customer behavior, and the various factors influencing pricing, fuel efficiency, and market share. This comprehensive review highlights critical findings, insights, and recommendations for companies to strengthen their position in the competitive labor market."""
    )
    st.write("______________")
    st.write("## Data Overview")
    st.write("The data used in this analysis includes the following columns:")
    st.write("- Brand: The manufacturer of the car.")
    st.write("- Model: The specific model of the car.")
    st.write("- Year: The year the car was manufactured.")
    st.write("- Price: The selling price of the car.")
    st.write("- Fuel Type: The type of fuel used by the car (e.g., petrol, diesel).")
    st.write("- Transmission: The type of transmission (e.g., manual, automatic).")
    st.write("- Mileage: The total mileage of the car.")
    st.write("- Tax: The tax amount associated with the car.")
    st.write("- MPG: Fuel efficiency of the car.")
    st.write("- Engine Size: The size of the car's engine.")
    st.write("The data includes a total of 4,960 rows and 10 columns.")
    st.write("______________")
    st.write("## Key Findings")
    st.write("- **Most Expensive Car**: Audi R8 (2020), semi-automatic, petrol.")
    st.write("- **Least Expensive Car**: Vauxhall Agila (2003), manual, petrol.")
    st.write("- **Highest Mileage**: Mercedes V-Class (2010), automatic, diesel.")
    st.write(
        "- **Highest Tax**: Ford Mustang (2017), BMW M3 (2009), Mercedes G-Class (2013), Mercedes M-Class (2014)."
    )
    st.write(
        "- **Price Trend**: Prices have risen over the years, with notable growth fluctuations and a peak around 2019."
    )
    st.write("- **Top-Selling Brands**: BMW and Mercedes-Benz dominate sales.")
    st.write(
        """- **Fuel Type Market Share**: 
        - Diesel (62.38%) leads, followed by Petrol (33.78%), with Hybrid and other types having smaller shares."""
    )
    st.write(
        """- **Transmission Types**: 
        - Semi-automatic has the highest sales, followed by automatic, while manual transmissions account for fewer sales."""
    )
    st.write(
        """- **Mileage and Transmission**: 
        - Manual cars generally have higher mileage, suggesting more use over long distances.
        - Cars with higher mileage tend to have lower prices, though there are some outliers."""
    )
    st.write(
        """- **Fuel Efficiency and Price**: 
        - Diesel cars have the highest average prices, followed by petrol.
        - There’s a general negative correlation between price and MPG, indicating pricier cars tend to have lower fuel efficiency, with some exceptions."""
    )
    st.write(
        "- **MPG Trend**: MPG has improved over time, but fluctuations exist due to external factors."
    )
    st.write(
        "- **Engine Size Trend**: Engine sizes have remained stable, with a slight increase in recent years."
    )
    st.write(
        """- **Engine Size and Fuel Type**: 
        - Diesel and petrol cars tend to have larger engines, while hybrid and electric cars have smaller engines.
        - Semi-automatic and manual transmissions exhibit a wider range of engine sizes compared to automatic."""
    )
    st.write(
        """- **Brand and Mileage**: 
        - Mileage distribution varies across brands, with Hyundai and Vauxhall showing lower mileage and Mercedes showing more variability."""
    )
    st.write(
        """- **Outliers**: Outliers are seen in multiple variables, especially for diesel and petrol cars, indicating extreme values in mileage, engine size, and price.
        - Different brands have distinct pricing and efficiency strategies, with variations in transmission and fuel efficiency."""
    )
    st.write("______________")
    st.write("## Analysis and Insights")
    st.write(
        "- **Price and Brand**: Audi's R8 is the most expensive car, while Vauxhall’s Agila is the least expensive, showing Audi's premium positioning versus Vauxhall’s affordability."
    )
    st.write(
        "- **Transmission and Fuel**: The most expensive car (semi-automatic, petrol) contrasts with the cheapest car (manual, petrol), indicating that both transmission and fuel type could impact price."
    )
    st.write(
        "- **Year and Mileage**: Newer cars (e.g., 2020 Audi) tend to be pricier, but high-mileage cars can still hold value, as seen with the 2010 Mercedes V-Class."
    )
    st.write(
        "- **Tax**: Cars with high tax (e.g., Mustang, M3) likely have larger engines and higher emissions."
    )
    st.write(
        "- **Price Trend**: Prices have been rising steadily, with a sharp peak around 2019, possibly due to economic factors or market shifts."
    )
    st.write(
        "- **Top Brands**: BMW and Mercedes-Benz dominate in sales, indicating strong brand preference."
    )
    st.write(
        "- **Pricing Strategy**: There is clear variation in pricing strategies across brands, with some focusing on premium offerings while others cater to budget-conscious consumers."
    )
    st.write(
        """- **Fuel Type**: 
        - Diesel dominates the market (62.38%), followed by petrol. Hybrid and electric cars have smaller shares, but this could grow due to technological advancements."""
    )
    st.write(
        """- **Transmission**: 
        - Semi-automatic transmissions lead in total sales, followed by automatic, while manual transmissions show higher mileage, suggesting different use cases."""
    )
    st.write(
        """- **Mileage and Price**: 
        - Higher mileage generally correlates with lower prices, but there are outliers where high-mileage cars maintain value."""
    )
    st.write(
        """- **Fuel Efficiency**: 
        - Diesel cars have better mileage than petrol, while hybrid and electric cars show moderate to low mileage, possibly due to battery range."""
    )
    st.write(
        "- **Engine Size**: Diesel and petrol cars typically have larger engines, while hybrid and electric vehicles focus on efficiency with smaller engines."
    )
    st.write(
        """- **Transmission and Engine Size**: 
        - Manual and semi-automatic cars have a wider range of engine sizes, reflecting their performance and flexibility in different driving conditions."""
    )
    st.write(
        """- **Driving Patterns**: Diesel cars are used more for long-distance driving, reflected in higher mileage. Meanwhile, manual transmissions appear more in cars driven longer distances."""
    )
    st.write(
        "- **Future Trends**: Advances in fuel efficiency and emissions regulations will likely continue to shape car designs and engine sizes, particularly in hybrid and electric vehicles."
    )
    st.write(
        "- **MPG and Price**: Cars with higher prices generally have lower MPG, emphasizing luxury and performance over fuel efficiency. However, this trend varies by brand and model."
    )
    st.write(
        "- **Brand Clusters**: Brands show distinct patterns in price, MPG, and market strategies, highlighting their unique target segments and customer bases."
    )
    st.write("see the data visualization for more insights")
    st.write("______________")
    st.write("## Recommendations")
    st.write(
        "- **Gather More Data**: Collect additional data, especially for earlier years, to better understand long-term trends."
    )
    st.write(
        "- **Analyze Other Factors**: Examine economic indicators, fuel prices, and technology advancements that may affect car prices."
    )
    st.write(
        "- **Compare to Industry Benchmarks**: Benchmark against competitors to assess the market position and pricing strategy."
    )
    st.write(
        "- **Forecast Future Prices**: Use time series analysis and forecasting models to predict future trends."
    )
    st.write(
        "- **Correlation Analysis**: Investigate correlations between variables such as price, brand, fuel type, transmission, engine size, and mileage to identify key drivers."
    )
    st.write(
        "- **Segmentation**: Break down data by car segment or model to uncover specific trends and patterns across different categories."
    )
    st.write(
        "- **Time Series Analysis**: Study sales and pricing trends over time to detect seasonal or cyclical fluctuations."
    )
    st.write(
        "- **Customer Segmentation**: Analyze customer behavior to identify preferences and target markets, optimizing marketing strategies accordingly."
    )
    st.write(
        "- **Monitor Industry Trends**: Keep an eye on trends and regulations affecting fuel efficiency, emissions standards, and new technologies."
    )
    st.write(
        "- **Explore Hybrid/Electric Vehicles**: Consider the potential of hybrid and electric vehicles for reducing emissions and improving fuel efficiency."
    )
    st.write(
        "- **Research & Development**: Invest in R&D to further enhance fuel efficiency without compromising performance."
    )
    st.write(
        "- **Engine and Transmission Trends**: Study the relationship between engine size, fuel type, and transmission to align offerings with market demand."
    )
    st.write(
        "- **Pricing Strategy**: Investigate how different car features, such as mileage and fuel efficiency, influence pricing and consumer preference."
    )
    st.write("______________")
    st.write("## Future Work")
    st.write(
        "- **Expand the Dataset**: Collect more data across different years, models, and segments to enhance analysis accuracy and robustness."
    )
    st.write(
        "- **Consider Additional Factors**: Include variables such as location, features, customer reviews, driving conditions, and government regulations that might impact car sales, engine size, fuel type, and mileage."
    )
    st.write(
        "- **Segmentation**: Investigate price trends and variations across different car segments or models."
    )
    st.write(
        "- **Correlation Analysis**: Explore relationships between price and other factors like engine size, mileage, transmission, and fuel type to uncover key drivers."
    )
    st.write(
        "- **Customer Segmentation**: Analyze customer preferences to identify potential pricing strategies and target markets."
    )
    st.write(
        "- **Advanced Analytics**: Apply advanced techniques like time series analysis and machine learning to extract deeper insights."
    )
    st.write(
        "- **Develop Predictive Models**: Build models to forecast trends in pricing, mileage, MPG, and customer preferences."
    )
    st.write(
        "- **Forecast Future Sales**: Use historical data and predictive models to project future sales trends and identify opportunities."
    )
    st.write("______________")
    st.write("# Conclusion")
    st.write("### Conclusion for Key Findings")
    st.write(
        """
    The analysis of the car sales data reveals several important trends that companies in the automotive sector should focus on. First, price variation is heavily influenced by the brand, model, and year of the car, with premium brands like Audi and BMW dominating the higher end of the price spectrum. On the other hand, more affordable options from brands like Vauxhall offer a counterbalance, indicating that brand perception plays a significant role in pricing.

    Transmission and fuel type are other key differentiators. Semi-automatic and diesel cars, which tend to offer better fuel efficiency and performance, are priced higher. However, the market is increasingly favoring hybrid and electric vehicles, which offer lower running costs and government incentives. Additionally, cars with lower mileage tend to command higher prices, yet high-mileage vehicles in excellent condition, such as well-maintained diesel cars, can still fetch significant value.

    The data also indicates that engine size and fuel efficiency are increasingly important factors for customers, with the trend leaning towards smaller engines, especially for hybrid and electric vehicles. This suggests a shift toward sustainability, reflecting consumer awareness of fuel efficiency and environmental impact."""
    )
    st.write("______________")
    st.write("### Conclusion for Analysis and Insights")
    st.write(
        """
    The analysis of the car sales dataset uncovers valuable insights into market trends. One notable finding is the steady increase in car prices over the past two decades, though the growth rate has fluctuated due to various factors such as economic conditions and regulatory changes. This suggests that the overall value of cars is rising, influenced by factors like technology, design improvements, and increased demand for fuel efficiency.

    The data also reveals that diesel cars dominate the market in terms of fuel efficiency and long-distance driving capability. However, the growing popularity of electric and hybrid vehicles signals a shift in consumer preference towards more sustainable options. Companies should be aware of this shift and consider investing in research and development to stay ahead in the transition to alternative energy vehicles.

    Transmission type also plays a significant role in pricing and customer preferences. Cars with manual transmissions tend to be more affordable, appealing to price-sensitive consumers, while automatic and semi-automatic transmissions are favored by those looking for convenience and a premium driving experience. Moreover, the relationship between transmission and engine size suggests that different types of cars, whether performance-oriented or budget-focused, cater to specific customer segments.

    Lastly, fuel efficiency continues to be a major factor driving customer decisions. Diesel cars tend to offer better mileage, but consumers are now gravitating toward hybrid and electric cars due to their environmental benefits and the availability of tax breaks or incentives. Companies should seize this opportunity to further explore fuel-efficient technologies and emphasize this in their marketing campaigns."""
    )
    st.write("______________")
    st.write("### Conclusion for Recommendations")
    st.write(
        """
    Based on the findings, automotive companies should adopt a multi-faceted strategy to stay competitive. Data-driven decision-making should be at the forefront of this strategy. Companies must continue gathering and analyzing extensive datasets, not only about pricing but also about external economic factors such as fuel prices, technological innovations, and government regulations. This will provide a holistic view of the market and allow for better strategic planning.

    Brand differentiation is another key area where companies can excel. Premium brands should focus on maintaining a reputation for luxury and high performance, while mid-range and budget brands might benefit from targeting a broader consumer base with affordable, fuel-efficient options. Companies should also consider expanding into the hybrid and electric vehicle markets, leveraging government incentives and appealing to eco-conscious consumers.

    Customer segmentation is critical to understanding diverse market needs. By analyzing customer behavior and preferences, companies can tailor marketing campaigns, product features, and pricing strategies to different customer groups. For example, luxury car buyers prioritize performance and aesthetics, while budget-conscious consumers look for fuel efficiency and affordability. Tailored marketing will help align the brand’s products with consumer expectations.

    Pricing strategies should reflect market trends. For brands with premium offerings, emphasizing performance and technology innovations will justify higher prices. Brands in the budget and mid-range segments should consider competitive pricing and promotional offers to attract price-sensitive customers. Additionally, businesses can explore introducing financing options to make higher-priced models more accessible."""
    )
    st.write("______________")
    st.write("### Conclusion for Future Work")
    st.write(
        """
    Moving forward, several strategies can help companies gain a competitive edge in the evolving automotive industry. Expanding the dataset is crucial for more accurate predictions. Data on newer models, evolving customer preferences, and additional economic factors such as maintenance costs, repair history, and financing options should be collected.

    Advanced analytics, such as machine learning models, can be used to predict future sales trends, allowing companies to identify market opportunities and adjust production accordingly. By implementing predictive models, businesses can forecast demand for specific models, fuel types, and transmission types, optimizing inventory and enhancing supply chain management.

    Time series analysis will provide deeper insights into cyclical trends and seasonal fluctuations in sales, enabling companies to plan marketing campaigns around peak sales periods. Additionally, correlation analysis can help identify the impact of new variables like government policies, emission standards, and even global economic shifts.

    Finally, companies should focus on continuous innovation. Investing in research and development, particularly in the realm of electric and hybrid vehicles, will keep businesses at the forefront of automotive technology. The transition to greener, more fuel-efficient cars is inevitable, and companies that adapt quickly to changing regulations and consumer demands will thrive."""
    )
    st.write("______________")
    st.write("### Conclusion of the Report")
    st.write(
        """
    Conclusion
    To remain competitive in a rapidly changing automotive market, companies must leverage the power of data analytics, focus on brand differentiation, and embrace innovation. Understanding customer behavior and preferences is key to targeting the right audience with the right product offerings. By implementing predictive analytics and expanding their focus on sustainable vehicle solutions, companies can secure their place in the future of the automotive industry.

    The key to long-term success lies in staying informed and agile, using data to drive decisions, and continuously innovating to meet consumer expectations. Companies that follow these steps will not only advance in the labor market but also position themselves as leaders in a highly competitive landscape."""
    )
    st.write("______________")
    st.write("## About")
    st.write(
        "#### This dashboard was created by [Eng. Mohamed Nasr](https://www.linkedin.com/in/medoox369)."
    )
    st.write(
        "#### The code for this dashboard is available on [GitHub](https://github.com/medoox369/Cars_sales/)"
    )
    st.write("______________")
    st.write("## End of Report")
    st.write("______________")
    st.write("## Thank You :smile:")


def predictions():
    import numpy as np
    import joblib
    import os
    import pandas as pd
    import streamlit as st
    from streamlit_option_menu import option_menu

    st.title("Car Price Predictor 🚗")
    original_data_file = "Cars sales.csv"
    if os.path.exists(original_data_file):
        df = pd.read_csv(original_data_file, encoding="ISO-8859-1")
        unique_brands = sorted(df["Brand"].unique())
        unique_models = sorted(df["Model"].unique())
    else:
        st.error("لم يتم العثور على ملف البيانات المعالجة 'Cars_sales.csv'!")
        st.stop()
    try:
        xgb_model = joblib.load("xgb_car_sales_model.pkl")
        encoder_brand = joblib.load("label_encoder_brand.pkl")
        encoder_model = joblib.load("label_encoder_model.pkl")
        scaler = joblib.load("scaler.pkl")
        selector = joblib.load("feature_selector.pkl")
    except FileNotFoundError as e:
        st.error(f"لم يتم العثور على ملف النموذج أو الترميزات: {e.filename}")
        st.stop()
    brand = st.selectbox("🔹Select Brand", unique_brands)
    model = st.selectbox("🔹Select Model", unique_models)
    year = st.slider("🔹Select Year", int(df["Year"].min()), int(df["Year"].max()))
    mileage = st.number_input("🔹Mileage", min_value=0, step=1)
    tax = st.number_input("🔹Tax", min_value=0, step=1)
    mpg = st.number_input("🔹MPG", min_value=0.0, step=0.1)
    engine_size = st.number_input("🔹Engine Size", min_value=0.0, step=0.1)
    if st.button("Predict Price 🚀"):
        try:
            new_car = pd.DataFrame(
                {
                    "Brand": [brand],
                    "Model": [model],
                    "Year": [year],
                    "Mileage": [mileage],
                    "Tax": [tax],
                    "MPG": [mpg],
                    "Engine Size": [engine_size],
                }
            )
            new_car["Brand"] = encoder_brand.transform(new_car["Brand"])
            new_car["Model"] = encoder_model.transform(new_car["Model"])
            numeric_columns = ["Year", "Mileage", "Tax", "MPG", "Engine Size"]
            new_car[numeric_columns] = scaler.transform(new_car[numeric_columns])
            new_car_selected = selector.transform(new_car)
            prediction = xgb_model.predict(new_car_selected)
            st.success(f"Predicted Price: {prediction[0]:,.2f} USD")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
    st.write("______________")
    st.write("## About")
    st.write(
        "#### This dashboard was created by [Eng. Mohamed Nasr](https://www.linkedin.com/in/medoox369)."
    )
    st.write(
        "#### The code for this dashboard is available on [GitHub](https://github.com/medoox369/Cars_sales/)"
    )
    st.write("______________")
    st.write("## Thank You :smile:")
    st.write("______________")


def contact():
    import streamlit as st
    import joblib
    from streamlit_option_menu import option_menu
    import os

    st.write("## About")
    st.write(
        "#### This dashboard was created by [Eng. Mohamed Nasr](https://www.linkedin.com/in/medoox369)."
    )
    st.write("#### The data used in this dashboard is from Eng. Mahmoud Tarek.")
    st.write(
        "#### The code for this dashboard is available on [GitHub](https://github.com/medoox369/Cars-sales)"
    )
    st.write("______________")
    st.write("## Contact")
    st.write(
        "#### For any inquiries, contact me via [WhatsApp](https://wa.me/+201276977748)."
    )
    st.write("______________")
    st.write("## Connect with me")
    st.write("### [LinkedIn](https://www.linkedin.com/in/medoox369)")
    st.write("### [GitHub](https://github.com/medoox369)")
    st.write("### [Email](mailto:https://medoox369@gmail.com)")
    st.write("### [Kaggle](https://www.kaggle.com/medoox369)")
    st.write("______________")
    st.write("## End of Page")
    st.write("______________")
    st.write("## Thank You :smile:")


# Main function to handle the menu and navigation
def streamlit_menu():
    selected = option_menu(
        menu_title=None,
        options=["Visualization", "Report", "Predictions", "Contact"],
        icons=["bar-chart-line-fill", "book", "graph-up-arrow", "envelope"],
        menu_icon="cast",
        orientation="horizontal",
    )
    return selected


# Initialize session state
if "selected" not in st.session_state:
    st.session_state["selected"] = None

selected = streamlit_menu()

if selected != st.session_state["selected"]:
    st.session_state["selected"] = selected
if st.session_state["selected"] == "Visualization":
    visualization()
elif st.session_state["selected"] == "Report":
    report()
elif st.session_state["selected"] == "Predictions":
    predictions()
elif st.session_state["selected"] == "Contact":
    contact()
