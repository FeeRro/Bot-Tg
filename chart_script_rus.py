import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# Define the components with better spacing and alignment
components = {
    # USER INTERFACE LAYER (Blue - #1FB8CD) - Layer 1
    'Telegram Bot\n(Русский интерфейс)': (3, 8.5, '#1FB8CD'),
    'Russian Cmd': (1, 7.5, '#1FB8CD'),
    'Voice Msg\n(Russian)': (5, 7.5, '#1FB8CD'),
    
    # LANGUAGE PROCESSING LAYER (Green - #2E8B57) - Layer 2
    'RU Text\nProcess': (0.5, 6, '#2E8B57'),
    'Speech Recog\n(Russian)': (2, 6, '#2E8B57'),
    'Translation\n(RU-EN)': (4, 6, '#2E8B57'),
    'Intent Class\n(Russian)': (5.5, 6, '#2E8B57'),
    
    # NEURAL NETWORK CORE (Red - #DB4545) - Layer 3
    'NumPy Neural\nNetwork': (3, 4.5, '#DB4545'),
    'RU Training\nData': (1, 4, '#DB4545'),
    'Intent Model': (5, 4, '#DB4545'),
    
    # EDUCATIONAL CONTENT (Yellow - #D2BA4C) - Layer 4
    'EN Lessons\n(RU explain)': (1, 2.5, '#D2BA4C'),
    'Grammar\n(Russian)': (2.5, 2.5, '#D2BA4C'),
    'Vocabulary\n(RU-EN)': (4, 2.5, '#D2BA4C'),
    'Progress\nTracking': (5.5, 2.5, '#D2BA4C'),
    
    # DATA STORAGE (Cyan - #5D878F) - Layer 5
    'Learning\nMaterials': (1.5, 1, '#5D878F'),
    'User\nProgress': (2.5, 1, '#5D878F'),
    'Language\nPatterns': (3.5, 1, '#5D878F'),
    'Analytics\nData': (4.5, 1, '#5D878F'),
}

# Define key connections (simplified to reduce clutter)
connections = [
    # UI to Processing (main flow)
    ('Telegram Bot\n(Русский интерфейс)', 'RU Text\nProcess'),
    ('Telegram Bot\n(Русский интерфейс)', 'Speech Recog\n(Russian)'),
    ('Russian Cmd', 'Intent Class\n(Russian)'),
    ('Voice Msg\n(Russian)', 'Speech Recog\n(Russian)'),
    
    # Processing to Neural Network
    ('RU Text\nProcess', 'NumPy Neural\nNetwork'),
    ('Speech Recog\n(Russian)', 'NumPy Neural\nNetwork'),
    ('Translation\n(RU-EN)', 'NumPy Neural\nNetwork'),
    ('Intent Class\n(Russian)', 'Intent Model'),
    
    # Neural Network to Content
    ('NumPy Neural\nNetwork', 'EN Lessons\n(RU explain)'),
    ('NumPy Neural\nNetwork', 'Grammar\n(Russian)'),
    ('Intent Model', 'Vocabulary\n(RU-EN)'),
    ('Intent Model', 'Progress\nTracking'),
    
    # Content to Storage
    ('EN Lessons\n(RU explain)', 'Learning\nMaterials'),
    ('Grammar\n(Russian)', 'Learning\nMaterials'),
    ('Vocabulary\n(RU-EN)', 'Language\nPatterns'),
    ('Progress\nTracking', 'User\nProgress'),
    ('Progress\nTracking', 'Analytics\nData'),
]

# Create the plot
fig = go.Figure()

# Add connections as lines first
for start, end in connections:
    start_pos = components[start]
    end_pos = components[end]
    
    fig.add_trace(go.Scatter(
        x=[start_pos[0], end_pos[0]],
        y=[start_pos[1], end_pos[1]],
        mode='lines',
        line=dict(color='lightgray', width=2),
        showlegend=False,
        hoverinfo='skip'
    ))

# Add components as scatter points
x_coords = []
y_coords = []
colors = []
labels = []

for component, (x, y, color) in components.items():
    x_coords.append(x)
    y_coords.append(y)
    colors.append(color)
    labels.append(component)

# Add scatter points for components with larger size
fig.add_trace(go.Scatter(
    x=x_coords,
    y=y_coords,
    mode='markers+text',
    marker=dict(
        size=80,
        color=colors,
        symbol='square',
        line=dict(width=3, color='white')
    ),
    text=labels,
    textposition='middle center',
    textfont=dict(size=10, color='white', family='Arial Black'),
    hoverinfo='text',
    hovertext=labels,
    showlegend=False
))

# Add layer background rectangles for better visual separation
layer_backgrounds = [
    # USER INTERFACE
    dict(type="rect", x0=0, y0=7, x1=6, y1=9, fillcolor='rgba(31, 184, 205, 0.1)', 
         line=dict(color='rgba(31, 184, 205, 0.3)', width=2)),
    # LANGUAGE PROCESSING  
    dict(type="rect", x0=0, y0=5.5, x1=6, y1=6.5, fillcolor='rgba(46, 139, 87, 0.1)', 
         line=dict(color='rgba(46, 139, 87, 0.3)', width=2)),
    # NEURAL NETWORK CORE
    dict(type="rect", x0=0, y0=3.5, x1=6, y1=5, fillcolor='rgba(219, 69, 69, 0.1)', 
         line=dict(color='rgba(219, 69, 69, 0.3)', width=2)),
    # EDUCATIONAL CONTENT
    dict(type="rect", x0=0, y0=2, x1=6, y1=3, fillcolor='rgba(210, 186, 76, 0.1)', 
         line=dict(color='rgba(210, 186, 76, 0.3)', width=2)),
    # DATA STORAGE
    dict(type="rect", x0=0, y0=0.5, x1=6, y1=1.5, fillcolor='rgba(93, 135, 143, 0.1)', 
         line=dict(color='rgba(93, 135, 143, 0.3)', width=2)),
]

# Add layer labels with better positioning
layer_labels = [
    ('USER INTERFACE', 0.2, 8.7, '#1FB8CD'),
    ('LANGUAGE PROCESSING', 0.2, 6.2, '#2E8B57'),
    ('NEURAL NETWORK CORE', 0.2, 4.7, '#DB4545'),
    ('EDUCATIONAL CONTENT', 0.2, 2.7, '#D2BA4C'),
    ('DATA STORAGE', 0.2, 1.2, '#5D878F')
]

for label, x, y, color in layer_labels:
    fig.add_trace(go.Scatter(
        x=[x], y=[y],
        mode='text',
        text=[label],
        textfont=dict(size=12, color=color, family='Arial Black'),
        showlegend=False,
        hoverinfo='skip'
    ))

# Update layout with better spacing and clean appearance
fig.update_layout(
    title='Russian Telegram Bot Architecture',
    xaxis=dict(
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        range=[-0.2, 6.2]
    ),
    yaxis=dict(
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        range=[0, 9.5]
    ),
    plot_bgcolor='white',
    showlegend=False,
    shapes=layer_backgrounds
)

# Save the chart
fig.write_image('telegram_bot_architecture.png')