import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_feature_impact():
    print("Loading pre-trained model and metadata...")
    try:
        model = joblib.load('heart_model.pkl')
        _, _, feature_names, _ = joblib.load('model_metadata.pkl')
    except FileNotFoundError:
        print("Model files not found! Please run train.py first.")
        return

    print("Calculating feature importance...")
    importances = model.feature_importances_
    
    # Create DataFrame for Visualization
    feature_imp_df = pd.DataFrame({
        'Feature': feature_names, 
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("Generating High-Resolution Visualization...")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette='magma', ax=ax)
    
    ax.set_title("Relative Impact of Clinical Features on Heart Disease Prediction", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Impact Score (Feature Importance)", fontsize=12)
    ax.set_ylabel("Clinical Feature", fontsize=12)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Save the plot as an image
    output_filename = 'feature_impact.png'
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    print(f"Success! Feature impact visualization has been saved as '{output_filename}'")

if __name__ == "__main__":
    generate_feature_impact()
