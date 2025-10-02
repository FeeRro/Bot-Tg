import plotly.graph_objects as go
import json

# Data from the provided JSON
data = {
  "components": [
    {"name": "Пользователь Telegram", "type": "input", "x": 50, "y": 50, "width": 150, "height": 60},
    {"name": "Telegram Bot API", "type": "interface", "x": 250, "y": 50, "width": 150, "height": 60},
    {"name": "Предобработка текста", "type": "processing", "x": 50, "y": 150, "width": 150, "height": 60},
    {"name": "Векторизация (Bag of Words)", "type": "processing", "x": 250, "y": 150, "width": 150, "height": 60},
    {"name": "Входной слой (vocab_size)", "type": "neural", "x": 50, "y": 250, "width": 150, "height": 60},
    {"name": "Скрытый слой (16 нейронов)", "type": "neural", "x": 250, "y": 250, "width": 150, "height": 60},
    {"name": "Выходной слой (4 класса)", "type": "neural", "x": 450, "y": 250, "width": 150, "height": 60},
    {"name": "Интенты: greeting, grammar, vocabulary, practice", "type": "output", "x": 450, "y": 150, "width": 200, "height": 60},
    {"name": "Генерация ответа", "type": "processing", "x": 450, "y": 50, "width": 150, "height": 60}
  ],
  "connections": [
    {"from": "Пользователь Telegram", "to": "Telegram Bot API"},
    {"from": "Telegram Bot API", "to": "Предобработка текста"},
    {"from": "Предобработка текста", "to": "Векторизация (Bag of Words)"},
    {"from": "Векторизация (Bag of Words)", "to": "Входной слой (vocab_size)"},
    {"from": "Входной слой (vocab_size)", "to": "Скрытый слой (16 нейронов)"},
    {"from": "Скрытый слой (16 нейронов)", "to": "Выходной слой (4 класса)"},
    {"from": "Выходной слой (4 класса)", "to": "Интенты: greeting, grammar, vocabulary, practice"},
    {"from": "Интенты: greeting, grammar, vocabulary, practice", "to": "Генерация ответа"},
    {"from": "Генерация ответа", "to": "Telegram Bot API"},
    {"from": "Telegram Bot API", "to": "Пользователь Telegram"}
  ]
}

# Improved color mapping for better grouping
colors = {
    "input": "#1FB8CD",      # User interface - cyan
    "interface": "#DB4545",   # API interface - red  
    "processing": "#2E8B57",  # Text processing - green
    "neural": "#5D878F",      # Neural network layers - consistent blue-gray
    "output": "#D2BA4C"       # Output/intents - yellow
}

# Create figure
fig = go.Figure()

# Improved layout with better spacing and alignment
improved_positions = {
    "Пользователь Telegram": {"x": 60, "y": 60, "width": 160, "height": 70},
    "Telegram Bot API": {"x": 270, "y": 60, "width": 160, "height": 70},
    "Генерация ответа": {"x": 480, "y": 60, "width": 160, "height": 70},
    "Предобработка текста": {"x": 60, "y": 170, "width": 160, "height": 70},
    "Векторизация (Bag of Words)": {"x": 270, "y": 170, "width": 160, "height": 70},
    "Интенты: greeting, grammar, vocabulary, practice": {"x": 480, "y": 170, "width": 160, "height": 70},
    "Входной слой (vocab_size)": {"x": 60, "y": 280, "width": 160, "height": 70},
    "Скрытый слой (16 нейронов)": {"x": 270, "y": 280, "width": 160, "height": 70},
    "Выходной слой (4 класса)": {"x": 480, "y": 280, "width": 160, "height": 70}
}

# Add rectangles and annotations
shapes = []
annotations = []

for comp in data["components"]:
    pos = improved_positions[comp["name"]]
    
    # Abbreviate names for better fit while keeping Russian
    display_name = comp["name"]
    if len(display_name) > 15:
        if "Пользователь" in display_name:
            display_name = "Польз. Telegram"
        elif "Предобработка" in display_name:
            display_name = "Предобр. текста"
        elif "Векторизация" in display_name:
            display_name = "Вектор. (BoW)"
        elif "Входной слой" in display_name:
            display_name = "Вход. слой"
        elif "Скрытый слой" in display_name:
            display_name = "Скр. слой (16)"
        elif "Выходной слой" in display_name:
            display_name = "Вых. слой (4)"
        elif "Интенты:" in display_name:
            display_name = "Интенты"
        elif "Генерация" in display_name:
            display_name = "Ген. ответа"
    
    shapes.append(
        dict(
            type="rect",
            x0=pos["x"],
            y0=400 - pos["y"] - pos["height"],
            x1=pos["x"] + pos["width"],
            y1=400 - pos["y"],
            fillcolor=colors[comp["type"]],
            line=dict(color="black", width=2),
            opacity=0.9
        )
    )
    
    # Add text annotations with better readability
    annotations.append(
        dict(
            x=pos["x"] + pos["width"]/2,
            y=400 - pos["y"] - pos["height"]/2,
            text=display_name,
            showarrow=False,
            font=dict(size=12, color="white", family="Arial Black"),
            xanchor="center",
            yanchor="middle"
        )
    )

# Create improved position mapping for connections
comp_positions = {}
for comp in data["components"]:
    pos = improved_positions[comp["name"]]
    comp_positions[comp["name"]] = {
        "x": pos["x"] + pos["width"]/2,
        "y": 400 - pos["y"] - pos["height"]/2,
        "width": pos["width"],
        "height": pos["height"]
    }

# Add improved arrows with better positioning
for conn in data["connections"]:
    from_comp = comp_positions[conn["from"]]
    to_comp = comp_positions[conn["to"]]
    
    # Calculate direction
    dx = to_comp["x"] - from_comp["x"]
    dy = to_comp["y"] - from_comp["y"]
    
    # Better edge connection points
    if abs(dx) > abs(dy):  # Horizontal connection
        if dx > 0:  # Moving right
            start_x = from_comp["x"] + from_comp["width"]/2 + 5
            end_x = to_comp["x"] - to_comp["width"]/2 - 5
        else:  # Moving left
            start_x = from_comp["x"] - from_comp["width"]/2 - 5
            end_x = to_comp["x"] + to_comp["width"]/2 + 5
        start_y = from_comp["y"]
        end_y = to_comp["y"]
    else:  # Vertical connection
        if dy > 0:  # Moving up
            start_y = from_comp["y"] + from_comp["height"]/2 + 5
            end_y = to_comp["y"] - to_comp["height"]/2 - 5
        else:  # Moving down
            start_y = from_comp["y"] - from_comp["height"]/2 - 5
            end_y = to_comp["y"] + to_comp["height"]/2 + 5
        start_x = from_comp["x"]
        end_x = to_comp["x"]
    
    # Add thicker arrow line
    fig.add_trace(go.Scatter(
        x=[start_x, end_x],
        y=[start_y, end_y],
        mode="lines",
        line=dict(color="black", width=3),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add larger arrowhead
    arrow_symbol = "triangle-right"
    if abs(dx) > abs(dy):
        arrow_symbol = "triangle-right" if dx > 0 else "triangle-left"
    else:
        arrow_symbol = "triangle-up" if dy > 0 else "triangle-down"
    
    fig.add_trace(go.Scatter(
        x=[end_x],
        y=[end_y],
        mode="markers",
        marker=dict(
            symbol=arrow_symbol,
            size=12,
            color="black"
        ),
        showlegend=False,
        hoverinfo='skip'
    ))

# Update layout with improved title and spacing
fig.update_layout(
    title="Chatbot Neural Net",
    shapes=shapes,
    annotations=annotations,
    xaxis=dict(range=[0, 700], showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(range=[0, 400], showgrid=False, zeroline=False, showticklabels=False),
    plot_bgcolor="white",
    showlegend=False
)

# Save the chart
fig.write_image("neural_network_architecture.png")