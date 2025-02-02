import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

file_path='Broker Compare.csv'
data=pd.read_csv(file_path)

data=data.iloc[1:].reset_index(drop=True)

selected_columns=['Broker Name', 'Equity Delhivery Brokerage(%)', 'Equity Intraday Brokerage', 'Equity Future Brokerage']
data2=data[selected_columns].copy()

numeric_columns = ['Equity Delhivery Brokerage(%)', 'Equity Intraday Brokerage', 'Equity Future Brokerage']
for col in numeric_columns:
    data2[col] = pd.to_numeric(data2[col], errors='coerce')
data2[numeric_columns] = data2[numeric_columns].fillna(0)

data2.loc[data2['Broker Name'] == 'Groww', numeric_columns] = [0.1, 0.05, 0.2]
data2['Broker Name'] = data2['Broker Name'].astype(str)

broker_names = data2['Broker Name']
equity_delivery = data2['Equity Delhivery Brokerage(%)']
equity_intraday = data2['Equity Intraday Brokerage']
equity_futures = data2['Equity Future Brokerage']

plt.figure(figsize=(10, 6))
plt.stackplot(
    broker_names,
    equity_delivery,
    equity_intraday,
    equity_futures,
    labels=['Equity Delivery', 'Equity Intraday', 'Equity Futures'],
    alpha=0.7
)

plt.legend(loc='upper left')
plt.title('Brokerage Charges Across Platforms')
plt.xlabel('Broker Name')
plt.ylabel('Charges')
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()


#########################################################################################################################################################



selected_columns2 = ['Broker Name','Downloads (M)']
data3 = data[selected_columns2].copy()

data3.loc[data3['Broker Name'] == 'Axis Direct', selected_columns2] = ['Axis Direct', 0.001]

# Convert to DataFrame
df2 = pd.DataFrame(data3)

# Extract data for the pie chart
broker_names = df2['Broker Name']
downloads = df2['Downloads (M)']
# percentages = (downloads / downloads.sum()) * 100
# Create a 2D Pie Chart
fig, ax = plt.subplots(figsize=(10, 8))  # Using a regular 2D axis
# ax.pie(
#     downloads,
#     labels=broker_names,
#     autopct='%1.1f%%',
#     startangle=140,
#     colors=plt.cm.tab20.colors  # Use a colormap for distinct colors
# )

wedges, texts, autotexts = ax.pie(
    downloads,
    autopct='%1.0f%%',
    startangle=140,
    colors=plt.cm.tab20.colors
)

# legend_labels = [f'{broker}: {percent:.1f}%' for broker, percent in zip(broker_names, percentages)]
# ax.legend(wedges, legend_labels, title="Brokers", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
ax.legend(wedges, broker_names, title="Brokers", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

# Add a title
plt.title('Downloads Distribution by Broker', fontsize=16)

# Show the plot
plt.show()


##################################################################################################################


ratings = data['Overall rating on Google'].tolist()

# Extract platform names
platforms = data['Broker Name'].tolist()

# Create an interactive bar plot using Plotly
fig = go.Figure(data=[go.Bar(
    x=platforms,  # Broker platforms on the x-axis
    y=ratings,    # Ratings on the y-axis
    text=ratings,  # Display ratings as hover text
    hoverinfo='x+y+text',  # Show both x (platform) and y (rating) values on hover
    marker=dict(color='teal', line=dict(color='black', width=1))  # Bar color and border styling
)])

# Customize the layout
fig.update_layout(
    title='Broker Platform Ratings on Google',
    xaxis_title='Broker Platforms',
    yaxis_title='Ratings',
    xaxis_tickangle=-45,  # Rotate x-axis labels for better readability
    template='plotly',
)

# Show the interactive plot
fig.show()

#########################################################################################################################################################



# Assuming the dataset has the following columns:
# 'Broker', 'Additional Features', 'Feature Count'
# If not, modify the column names below accordingly.

# Ensure 'Additional Features' column has no NaN values and is treated as a string
df=data
df['Additional Features'] = df['Additional Features'].fillna('').astype(str)

# Preprocess the data: Calculate feature count for each broker
df['Feature Count'] = df['Additional Features'].str.split(',').apply(len) 

# Create the bubble chart
fig = px.scatter(
    df,
    x='Feature Count',  # Replace with your x-axis variable
    y='Broker Name',         # Replace with your y-axis variable
    size='Feature Count',  # Size of the bubble
    color='Broker Name',        # Color bubbles by broker
    hover_data={'Additional Features': True},  # Show additional features on hover
    title="Bubble Chart of Brokers and Their Features",
    labels={"Feature Count": "Number of Features", "Broker Name": "Broker Name"},
    template="plotly_white"
)

# Customize layout
fig.update_traces(marker=dict(opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
fig.update_layout(showlegend=False)

fig.update_layout(
    width=1500,  # Set the width of the plot
    height=700,
    xaxis=dict(
        showline=True,         # Show x-axis line
        linecolor='black',     # Set the color of the x-axis line
        linewidth=1,           # Set the width of the x-axis line
        title="Number of Features",  # Label for the x-axis
        dtick=0 
    ),
    yaxis=dict(
        showline=True,         # Show y-axis line
        linecolor='black',     # Set the color of the y-axis line
        linewidth=2,           # Set the width of the y-axis line
        title="Broker Name"    # Label for the y-axis
    )
    
)

# Show the chart
fig.show()

#############################################################################################################################################################################################