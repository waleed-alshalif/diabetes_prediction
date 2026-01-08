import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Import only standard libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, recall_score, f1_score, 
                           confusion_matrix, classification_report, 
                           roc_auc_score, roc_curve, auc)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_and_clean_data(filepath):
    """Load and clean diabetes data"""
    data = pd.read_csv(filepath)
    
    # Handle missing values (0 in medical data often means missing)
    cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_to_fix:
        data[col] = data[col].replace(0, data[col].median())
    
    return data

def explore_data(data):
    """Basic data exploration"""
    print("Dataset Shape:", data.shape)
    print("\nClass Distribution:")
    print(data['Outcome'].value_counts())
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.show()

def train_models(X, y):
    """Train and evaluate multiple models"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models with class balancing
    models = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'RandomForestClassifier': RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced'),
        'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42, n_estimators=100),
        'SVC': SVC(random_state=42, probability=True, class_weight='balanced')
    }
    
    # Lists to store results
    algorithm_names = []
    accuracy_results = []
    recall_results = []
    f1_results = []
    confusion_matrices = []
    
    print("\n" + "="*60)
    print("TRAINING AND EVALUATING MODELS")
    print("="*60)
    
    for name, model in models.items():
        print(f"\n{'='*40}")
        print(f"Training {name}")
        print('='*40)
        
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Get confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Store results in lists
        algorithm_names.append(name)
        accuracy_results.append(accuracy)
        recall_results.append(recall)
        f1_results.append(f1)
        confusion_matrices.append(cm)
        
        # Print results
        print("Confusion Matrix:")
        print(cm)
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Get probabilities for ROC if available
        if hasattr(model, 'predict_proba'):
            y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_prob)
            print(f"ROC-AUC: {roc_auc:.4f}")
    
    # Create final report DataFrame
    final_report = pd.DataFrame({
        "Algorithm": algorithm_names,
        "Accuracy": accuracy_results,
        "Recall": recall_results,
        "F1_Score": f1_results
    })
    
    # Add confusion matrices as string representation
    confusion_strings = []
    for cm in confusion_matrices:
        confusion_strings.append(f"TN={cm[0,0]}, FP={cm[0,1]}\nFN={cm[1,0]}, TP={cm[1,1]}")
    
    final_report["Confusion_Matrix"] = confusion_strings
    
    print("\n" + "="*60)
    print("FINAL RESULTS COMPARISON")
    print("="*60)
    print(final_report.to_string(index=False))
    
    return final_report, algorithm_names, accuracy_results, recall_results, f1_results

def plot_results(algorithm_names, accuracy_results, recall_results, f1_results):
    """Plot the results comparison"""
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Bar plot for accuracy
    axes[0, 0].bar(algorithm_names, accuracy_results, color='skyblue', alpha=0.8)
    axes[0, 0].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Add values on top of bars
    for i, v in enumerate(accuracy_results):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
    
    # Bar plot for recall
    axes[0, 1].bar(algorithm_names, recall_results, color='lightcoral', alpha=0.8)
    axes[0, 1].set_title('Recall Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Add values on top of bars
    for i, v in enumerate(recall_results):
        axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
    
    # Bar plot for F1-Score
    axes[1, 0].bar(algorithm_names, f1_results, color='lightgreen', alpha=0.8)
    axes[1, 0].set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Add values on top of bars
    for i, v in enumerate(f1_results):
        axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
    
    # Line plot for all metrics
    x_positions = np.arange(len(algorithm_names))
    axes[1, 1].plot(x_positions, accuracy_results, marker='o', label='Accuracy', linewidth=2)
    axes[1, 1].plot(x_positions, recall_results, marker='s', label='Recall', linewidth=2)
    axes[1, 1].plot(x_positions, f1_results, marker='^', label='F1-Score', linewidth=2)
    axes[1, 1].set_title('All Metrics Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Algorithms')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_xticks(x_positions)
    axes[1, 1].set_xticklabels(algorithm_names, rotation=45)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend(loc='best')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function"""
    print("DIABETES PREDICTION PROJECT")
    print("="*60)
    
    try:
        # Load data
        data = load_and_clean_data("Datasets/diabetes.csv")
        
        # Explore
        explore_data(data)
        
        # Prepare features
        X = data.drop('Outcome', axis=1)
        y = data['Outcome']
        
        # Train models and get results
        final_report, algorithm_names, accuracy_results, recall_results, f1_results = train_models(X, y)
        
        # Plot results
        plot_results(algorithm_names, accuracy_results, recall_results, f1_results)
        
        # Save results to CSV
        final_report.to_csv('diabetes_model_results.csv', index=False)
        print("\nResults saved to 'diabetes_model_results.csv'")
        
        print("\n" + "="*60)
        print("PROJECT COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Display the final report again
        print("\nFINAL REPORT SUMMARY:")
        print(final_report[['Algorithm', 'Accuracy', 'Recall', 'F1_Score']].to_string(index=False))

        print("\n\n")
        print("Waleed abdullatif ahmed alshalif")
        print("walshalif@gmail.com")
        

    except FileNotFoundError:
        print("ERROR: 'Datasets/diabetes.csv' file not found!")
        print("Please make sure the file exists in the Datasets folder.")
    
    except Exception as e:
        print(f"ERROR: {str(e)}")
        print("Please check your data and try again.")