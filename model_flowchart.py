import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_flowchart():
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Function to draw a box with text
    def draw_box(x, y, width, height, text, color='lightblue', edge_color='black'):
        rect = patches.FancyBboxPatch((x - width/2, y - height/2), width, height,
                                      boxstyle="round,pad=0.02",
                                      linewidth=1.5, edgecolor=edge_color, facecolor=color)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')
        return (x, y)

    # Function to draw an arrow
    def draw_arrow(start, end):
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="->", lw=1.5, color='black'))

    # --- Nodes ---

    # Level 0: Input
    input_pos = draw_box(0.5, 0.95, 0.3, 0.05, "Input Data\n(Dengue.xlsx)", color='#e1f5fe')

    # Level 1: Preprocessing
    prep_pos = draw_box(0.5, 0.85, 0.4, 0.06, "Preprocessing\n(Scaling, Lag Features, Train/Test Split)", color='#fff9c4')

    # Level 2: Base Models (Parallel)
    # We have 4 models. Let's space them out.
    y_models = 0.65
    lstm_pos = draw_box(0.2, y_models, 0.15, 0.05, "LSTM\n(Deep Learning)", color='#ffccbc')
    gru_pos = draw_box(0.4, y_models, 0.15, 0.05, "GRU\n(Deep Learning)", color='#ffccbc')
    rnn_pos = draw_box(0.6, y_models, 0.15, 0.05, "SimpleRNN\n(Deep Learning)", color='#ffccbc')
    prophet_pos = draw_box(0.8, y_models, 0.15, 0.05, "Prophet\n(Statistical)", color='#c8e6c9')

    # Level 3: Intermediate Predictions
    y_preds = 0.50
    preds_pos = draw_box(0.5, y_preds, 0.6, 0.05, "Base Model Predictions\n(Stacked as Meta-Features)", color='#e0e0e0')

    # Level 4: Meta Learner
    y_meta = 0.35
    meta_pos = draw_box(0.5, y_meta, 0.3, 0.05, "Meta-Learner\n(XGBoost Regressor)", color='#d1c4e9')

    # Level 5: Output
    y_out = 0.20
    output_pos = draw_box(0.5, y_out, 0.3, 0.05, "Final Ensemble Forecast\n(Risk Assessment & Planning)", color='#b2dfdb')

    # --- Edges ---

    # Input -> Preprocessing
    draw_arrow((input_pos[0], input_pos[1] - 0.025), (prep_pos[0], prep_pos[1] + 0.03))

    # Preprocessing -> Base Models
    # Deep Learning models take the same preprocessed data
    draw_arrow((prep_pos[0], prep_pos[1] - 0.03), (lstm_pos[0], lstm_pos[1] + 0.025))
    draw_arrow((prep_pos[0], prep_pos[1] - 0.03), (gru_pos[0], gru_pos[1] + 0.025))
    draw_arrow((prep_pos[0], prep_pos[1] - 0.03), (rnn_pos[0], rnn_pos[1] + 0.025))
    
    # Prophet usually takes slightly different format (ds, y), but comes from same source
    draw_arrow((prep_pos[0], prep_pos[1] - 0.03), (prophet_pos[0], prophet_pos[1] + 0.025))

    # Base Models -> Predictions
    draw_arrow((lstm_pos[0], lstm_pos[1] - 0.025), (preds_pos[0] - 0.2, preds_pos[1] + 0.025))
    draw_arrow((gru_pos[0], gru_pos[1] - 0.025), (preds_pos[0] - 0.07, preds_pos[1] + 0.025))
    draw_arrow((rnn_pos[0], rnn_pos[1] - 0.025), (preds_pos[0] + 0.07, preds_pos[1] + 0.025))
    draw_arrow((prophet_pos[0], prophet_pos[1] - 0.025), (preds_pos[0] + 0.2, preds_pos[1] + 0.025))

    # Predictions -> Meta Learner
    draw_arrow((preds_pos[0], preds_pos[1] - 0.025), (meta_pos[0], meta_pos[1] + 0.025))

    # Meta Learner -> Output
    draw_arrow((meta_pos[0], meta_pos[1] - 0.025), (output_pos[0], output_pos[1] + 0.025))

    # Add Title
    plt.title("Stacked Ensemble Model Architecture for Dengue Forecasting", fontsize=16, fontweight='bold', pad=20)
    
    # Save
    plt.tight_layout()
    plt.savefig('model_architecture_flowchart.png', dpi=300, bbox_inches='tight')
    print("Flowchart saved as 'model_architecture_flowchart.png'")
    plt.close()

if __name__ == "__main__":
    draw_flowchart()
