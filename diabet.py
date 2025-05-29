import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.inspection import permutation_importance
import joblib
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from IPython.display import display, HTML

# Suppress warnings
warnings.simplefilter(action="ignore")

# Setting visualization style
plt.style.use('fivethirtyeight')
sns.set_palette("viridis")

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Create a class to handle the entire process for better organization
class DiabetesPredictionAnalysis:
    def __init__(self, data_path="diabetes.csv"):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.model = None
        self.scaler = None
        self.medians = {}
        self.feature_importance = None
        self.cv_scores = None

    def print_section_header(self, title):
        """Print a formatted section header"""
        print("\n" + "="*80)
        print(f" {title} ".center(80, "="))
        print("="*80 + "\n")

    def load_data(self):
        """Load the diabetes dataset"""
        self.print_section_header("Loading and Exploring Data")
        start_time = time.time()

        print("📊 Loading diabetes dataset...")
        self.df = pd.read_csv(self.data_path)

        print(f"✅ Data loaded successfully! Shape: {self.df.shape}")
        print(f"⏱️ Time taken: {time.time() - start_time:.2f} seconds")

        # Display basic information about the dataset
        print("\n📋 First 5 rows of the dataset:")
        display(self.df.head())

        print("\n📊 Dataset information:")
        display(pd.DataFrame({
            "Column": self.df.columns,
            "Data Type": self.df.dtypes,
            "Non-Null Count": self.df.count(),
            "Null Count": self.df.isnull().sum(),
            "Unique Values": [self.df[col].nunique() for col in self.df.columns],
            "Min Value": [self.df[col].min() for col in self.df.columns],
            "Max Value": [self.df[col].max() for col in self.df.columns]
        }))

        # Display summary statistics
        print("\n📊 Summary statistics:")
        display(self.df.describe().T)

        return self

    def visualize_data_distributions(self):
        """Visualize distributions of variables in the dataset"""
        self.print_section_header("Data Distribution Analysis")

        # Outcome distribution
        print("🔍 Analyzing outcome distribution...")
        outcome_counts = self.df['Outcome'].value_counts()

        plt.figure(figsize=(10, 6))
        sns.countplot(x='Outcome', data=self.df, palette=['#2ecc71', '#e74c3c'])
        plt.title('Distribution of Diabetes Outcome (0: No Diabetes, 1: Diabetes)', fontsize=15)
        plt.xlabel('Outcome', fontsize=12)
        plt.ylabel('Count', fontsize=12)

        # Add percentage labels
        total = len(self.df)
        for i, count in enumerate(outcome_counts):
            percentage = count / total * 100
            plt.annotate(f'{count} ({percentage:.1f}%)',
                        (i, count),
                        ha='center',
                        va='bottom',
                        fontsize=12)
        plt.show()

        # Distribution of numeric features
        print("\n🔍 Analyzing feature distributions...")

        numeric_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

        fig, axes = plt.subplots(4, 2, figsize=(18, 20))
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
            # Histogram with KDE
            sns.histplot(data=self.df, x=col, hue='Outcome', kde=True, ax=axes[i],
                         palette=['#2ecc71', '#e74c3c'], alpha=0.6, bins=25)
            axes[i].set_title(f'Distribution of {col} by Diabetes Outcome', fontsize=14)
            axes[i].set_xlabel(col, fontsize=12)
            axes[i].set_ylabel('Count', fontsize=12)
            axes[i].legend(['No Diabetes', 'Diabetes'])

        plt.tight_layout()
        plt.show()

        # Boxplots to identify potential outliers
        print("\n🔍 Identifying potential outliers using boxplots...")

        fig, axes = plt.subplots(4, 2, figsize=(18, 20))
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
            sns.boxplot(x='Outcome', y=col, data=self.df, ax=axes[i], palette=['#2ecc71', '#e74c3c'])
            axes[i].set_title(f'Boxplot of {col} by Diabetes Outcome', fontsize=14)
            axes[i].set_xlabel('Outcome (0: No Diabetes, 1: Diabetes)', fontsize=12)
            axes[i].set_ylabel(col, fontsize=12)

        plt.tight_layout()
        plt.show()

        # Check for zero values that should be missing
        print("\n🔍 Analyzing zero values that might represent missing data...")
        zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

        zero_counts = {col: (self.df[col] == 0).sum() for col in zero_cols}
        zero_percentages = {col: (self.df[col] == 0).mean() * 100 for col in zero_cols}

        zero_df = pd.DataFrame({
            'Column': zero_cols,
            'Zero Count': [zero_counts[col] for col in zero_cols],
            'Zero Percentage': [zero_percentages[col] for col in zero_cols]
        })

        display(zero_df)

        # Visualize zero value counts
        plt.figure(figsize=(12, 6))
        bars = plt.bar(zero_df['Column'], zero_df['Zero Count'], color='#3498db')
        plt.title('Count of Zero Values in Features (Potential Missing Data)', fontsize=15)
        plt.xlabel('Feature', fontsize=12)
        plt.ylabel('Count of Zeros', fontsize=12)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height}', ha='center', va='bottom', fontsize=12)

        plt.tight_layout()
        plt.show()

        return self

    def analyze_correlations(self):
        """Analyze correlations between variables"""
        self.print_section_header("Correlation Analysis")

        print("🔍 Analyzing correlations between features...")

        # Calculate correlation matrix
        corr_matrix = self.df.corr()

        # Plot heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, annot=True, fmt=".2f",
                   linewidths=0.5, center=0, square=True, vmin=-1, vmax=1)
        plt.title('Correlation Matrix of Features', fontsize=16)
        plt.tight_layout()
        plt.show()

        # Plot correlation with target
        target_corr = corr_matrix['Outcome'].drop('Outcome').sort_values(ascending=False)

        plt.figure(figsize=(12, 8))
        bars = plt.bar(target_corr.index, target_corr.values, color=plt.cm.viridis(np.linspace(0, 1, len(target_corr))))
        plt.title('Correlation of Features with Diabetes Outcome', fontsize=16)
        plt.xlabel('Feature', fontsize=12)
        plt.ylabel('Correlation Coefficient', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.,
                    height + 0.01 if height >= 0 else height - 0.03,
                    f'{height:.2f}',
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=10)

        plt.tight_layout()
        plt.show()

        # Pair plot for the most correlated features
        print("\n🔍 Creating pair plot for the top correlated features with Outcome...")
        top_features = target_corr.abs().nlargest(3).index.tolist()
        top_features.append('Outcome')

        sns.pairplot(self.df[top_features], hue='Outcome', palette=['#2ecc71', '#e74c3c'],
                    diag_kind='kde', plot_kws={'alpha': 0.6})
        plt.suptitle('Pair Plot of Top Correlated Features', y=1.02, fontsize=16)
        plt.tight_layout()
        plt.show()

        return self

    def preprocess_data(self):
        """Preprocess the data by handling missing values and normalization"""
        self.print_section_header("Data Preprocessing")
        start_time = time.time()

        print("🔧 Handling missing values (zeros in medical features)...")

        # Get columns that shouldn't have zeros (medically unlikely)
        zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']

        # Replace zeros with NaN for these columns
        for col in zero_columns:
            original_zeros = (self.df[col] == 0).sum()
            self.df[col] = np.where(self.df[col] == 0, np.nan, self.df[col])
            current_nans = self.df[col].isna().sum()
            print(f"  - {col}: Replaced {original_zeros} zeros with NaN")

        # Calculate and store median values for each column
        print("\n🔧 Calculating median values for imputation...")
        for col in zero_columns:
            self.medians[col] = self.df[col].median()
            print(f"  - {col} median: {self.medians[col]:.2f}")

        # Fill missing values with calculated medians
        print("\n🔧 Imputing missing values with medians...")
        for col in zero_columns:
            self.df[col].fillna(self.medians[col], inplace=True)

        # Save medians for prediction
        joblib.dump(self.medians, "diabetes_medians.pkl")
        print("✅ Saved median values to 'diabetes_medians.pkl'")

        # Split data into features and target
        print("\n🔧 Splitting data into features and target...")
        X = self.df.drop("Outcome", axis=1)
        y = self.df["Outcome"]

        # Split data into training and testing sets
        print("\n🔧 Splitting data into training and testing sets...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"  - Training set: {self.X_train.shape[0]} samples")
        print(f"  - Testing set: {self.X_test.shape[0]} samples")

        # Check class distribution in splits
        train_dist = pd.Series(self.y_train).value_counts(normalize=True).to_dict()
        test_dist = pd.Series(self.y_test).value_counts(normalize=True).to_dict()

        print(f"  - Training set class distribution: {train_dist}")
        print(f"  - Testing set class distribution: {test_dist}")

        # Standardize features
        print("\n🔧 Standardizing features...")
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        # Save the scaler
        joblib.dump(self.scaler, "diabetes_scaler.pkl")
        print("✅ Saved scaler to 'diabetes_scaler.pkl'")

        print(f"\n⏱️ Preprocessing completed in {time.time() - start_time:.2f} seconds")

        return self

    def feature_selection_analysis(self):
        """Analyze feature importance for feature selection"""
        self.print_section_header("Feature Selection Analysis")

        print("🔍 Training a preliminary model for feature importance...")

        # Train a preliminary Random Forest model
        prelim_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        prelim_model.fit(self.X_train_scaled, self.y_train)

        # Get feature importance
        importance = prelim_model.feature_importances_
        feature_names = self.X_train.columns

        # Create DataFrame for better visualization
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)

        print("\n📊 Feature importance from preliminary Random Forest model:")
        display(feature_importance_df)

        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
        plt.title('Feature Importance from Random Forest', fontsize=16)
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        plt.show()

        # Perform permutation importance (more reliable)
        print("\n🔍 Calculating permutation importance (more reliable)...")

        perm_importance = permutation_importance(
            prelim_model, self.X_test_scaled, self.y_test,
            n_repeats=10, random_state=42, n_jobs=-1
        )

        # Create DataFrame for permutation importance
        perm_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': perm_importance.importances_mean,
            'Std': perm_importance.importances_std
        }).sort_values('Importance', ascending=False)

        print("\n📊 Permutation importance results:")
        display(perm_importance_df)

        # Plot permutation importance
        plt.figure(figsize=(12, 8))
        plt.errorbar(
            x=perm_importance_df['Importance'],
            y=range(len(perm_importance_df)),
            xerr=perm_importance_df['Std'],
            fmt='o',
            capsize=5
        )
        plt.yticks(range(len(perm_importance_df)), perm_importance_df['Feature'])
        plt.title('Permutation Feature Importance with Standard Deviation', fontsize=16)
        plt.xlabel('Mean Decrease in Accuracy', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

        return self

    def train_model(self):
        """Train the diabetes prediction model"""
        self.print_section_header("Model Training")
        start_time = time.time()

        print("🧠 Training Random Forest model with improved hyperparameters...")

        # Define the Random Forest model with optimized hyperparameters
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',  # Handle class imbalance
            random_state=42,
            n_jobs=-1  # Use all available cores
        )

        # Train the model
        self.model.fit(self.X_train_scaled, self.y_train)

        # Save the trained model
        joblib.dump(self.model, "diabetes_model.pkl")
        print("✅ Saved model to 'diabetes_model.pkl'")

        # Perform cross-validation to get a better estimate of model performance
        print("\n🔍 Performing 5-fold cross-validation...")

        cv_scores = cross_val_score(
            self.model, self.X_train_scaled, self.y_train,
            cv=5, scoring='accuracy', n_jobs=-1
        )

        self.cv_scores = cv_scores
        print(f"  - Cross-validation scores: {cv_scores}")
        print(f"  - Mean CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        print(f"\n⏱️ Model training completed in {time.time() - start_time:.2f} seconds")

        return self

    def hyperparameter_tuning(self, quick=True):
        """Perform hyperparameter tuning to find the best model"""
        self.print_section_header("Hyperparameter Tuning")
        start_time = time.time()

        if quick:
            print("🔧 Performing quick hyperparameter tuning...")
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [None, 10],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        else:
            print("🔧 Performing comprehensive hyperparameter tuning (this may take a while)...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 5, 10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False],
                'class_weight': [None, 'balanced']
            }

        # Create a GridSearchCV object
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid=param_grid,
            cv=5,
            scoring='f1',  # F1 score is good for imbalanced data
            n_jobs=-1,
            verbose=1
        )

        # Fit the grid search
        grid_search.fit(self.X_train_scaled, self.y_train)

        # Get the best parameters and best score
        print(f"\n✅ Best parameters found: {grid_search.best_params_}")
        print(f"✅ Best cross-validation score: {grid_search.best_score_:.4f}")

        # Update model with best parameters
        self.model = grid_search.best_estimator_

        # Save the tuned model
        joblib.dump(self.model, "diabetes_model_tuned.pkl")
        print("✅ Saved tuned model to 'diabetes_model_tuned.pkl'")

        print(f"\n⏱️ Hyperparameter tuning completed in {time.time() - start_time:.2f} seconds")

        return self

    def evaluate_model(self):
        """Evaluate the trained model"""
        self.print_section_header("Model Evaluation")

        print("🔍 Making predictions on test set...")
        y_pred = self.model.predict(self.X_test_scaled)
        y_pred_proba = self.model.predict_proba(self.X_test_scaled)[:, 1]

        # Calculate performance metrics
        print("\n📊 Model performance metrics:")
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        auc_score = roc_auc_score(self.y_test, y_pred_proba)

        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'],
            'Value': [accuracy, precision, recall, f1, auc_score]
        })

        display(metrics_df)

        # Confusion matrix
        print("\n📊 Confusion matrix:")
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Diabetes', 'Diabetes'],
                   yticklabels=['No Diabetes', 'Diabetes'])
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix', fontsize=16)
        plt.tight_layout()
        plt.show()

        # Calculate confusion matrix metrics
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)

        print(f"\n📊 Additional metrics:")
        print(f"  - True Positives (TP): {tp}")
        print(f"  - False Positives (FP): {fp}")
        print(f"  - True Negatives (TN): {tn}")
        print(f"  - False Negatives (FN): {fn}")
        print(f"  - Sensitivity/Recall: {recall:.4f}")
        print(f"  - Specificity: {specificity:.4f}")
        print(f"  - Precision: {precision:.4f}")

        # Classification report
        print("\n📊 Classification report:")
        report = classification_report(self.y_test, y_pred, target_names=['No Diabetes', 'Diabetes'])
        print(report)

        # ROC curve
        print("\n📊 ROC curve:")
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
        plt.legend(loc="lower right")
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

        # Feature importance from the final model
        print("\n📊 Feature importance from the final model:")
        feature_importance = pd.DataFrame({
            'Feature': self.X_train.columns,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)

        self.feature_importance = feature_importance
        display(feature_importance)

        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
        plt.title('Feature Importance from Final Model', fontsize=16)
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        plt.show()

        return self

    def analyze_misclassifications(self):
        """Analyze misclassified instances"""
        self.print_section_header("Misclassification Analysis")

        print("🔍 Analyzing misclassified instances...")

        # Get predictions
        y_pred = self.model.predict(self.X_test_scaled)
        y_pred_proba = self.model.predict_proba(self.X_test_scaled)[:, 1]

        # Identify misclassified instances
        misclassified = self.X_test.copy()
        misclassified['Actual'] = self.y_test.values
        misclassified['Predicted'] = y_pred
        misclassified['Probability'] = y_pred_proba
        misclassified['Misclassified'] = misclassified['Actual'] != misclassified['Predicted']

        # Filter misclassified instances
        misclassified_df = misclassified[misclassified['Misclassified']]

        print(f"\n📊 Number of misclassified instances: {len(misclassified_df)} out of {len(self.X_test)} test instances")
        print(f"📊 Misclassification rate: {len(misclassified_df) / len(self.X_test):.4f}")

        if len(misclassified_df) > 0:
            # Display misclassified instances
            print("\n📊 Sample of misclassified instances:")
            display(misclassified_df.head())

            # Analyze misclassifications by class
            false_positives = misclassified_df[misclassified_df['Actual'] == 0]
            false_negatives = misclassified_df[misclassified_df['Actual'] == 1]

            print(f"\n📊 False Positives (predicted diabetes, but actually no diabetes): {len(false_positives)}")
            print(f"📊 False Negatives (predicted no diabetes, but actually has diabetes): {len(false_negatives)}")

            # Compare feature distributions between correctly and incorrectly classified instances
            print("\n🔍 Comparing feature distributions between correctly and incorrectly classified instances...")

            for col in self.X_test.columns:
                plt.figure(figsize=(12, 6))

                # Create a more detailed DataFrame for plotting
                plot_df = misclassified.copy()
                plot_df['Classification'] = 'Correctly Classified'
                plot_df.loc[plot_df['Misclassified'], 'Classification'] = 'Misclassified'

                # Split by actual class
                plot_df_0 = plot_df[plot_df['Actual'] == 0]
                plot_df_1 = plot_df[plot_df['Actual'] == 1]

                # Create subplots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

                # Plot for Actual Class 0
                sns.boxplot(x='Classification', y=col, data=plot_df_0, ax=ax1, palette=['#2ecc71', '#e74c3c'])
                ax1.set_title(f'{col} Distribution for Actual Class 0 (No Diabetes)', fontsize=14)
                ax1.set_xlabel('Classification Result', fontsize=12)
                ax1.set_ylabel(col, fontsize=12)

                # Plot for Actual Class 1
                sns.boxplot(x='Classification', y=col, data=plot_df_1, ax=ax2, palette=['#2ecc71', '#e74c3c'])
                ax2.set_title(f'{col} Distribution for Actual Class 1 (Diabetes)', fontsize=14)
                ax2.set_xlabel('Classification Result', fontsize=12)
                ax2.set_ylabel(col, fontsize=12)

                plt.tight_layout()
                plt.show()
        else:
            print("✅ No misclassified instances found!")

        return self

    def create_prediction_function(self):
        """Create and save the prediction function"""
        self.print_section_header("Creating Prediction Function")

        print("📝 Creating prediction function file...")

        with open("predict_diabetes.py", "w") as f:
            f.write("""
import numpy as np
import pandas as pd
import joblib

def predict_diabetes(data):
    \"\"\"
    Predict diabetes for a given set of features

    Parameters:
    data (dict): Dictionary with keys - 'Pregnancies', 'Glucose', 'BloodPressure',
                'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'

    Returns:
    dict: Dictionary with prediction results
    \"\"\"
    # Load model and preprocessing components
    model = joblib.load("diabetes_model.pkl")
    scaler = joblib.load("diabetes_scaler.pkl")
    medians = joblib.load("diabetes_medians.pkl")

    # Convert input to DataFrame
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = pd.DataFrame([data], columns=['Pregnancies', 'Glucose', 'BloodPressure',
                                         'SkinThickness', 'Insulin
            , 'BMI',
                                         'DiabetesPedigreeFunction', 'Age'])

    # Handle missing/zero values
    zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']
    for col in zero_columns:
        df[col] = np.where(df[col] == 0, np.nan, df[col])
        df[col].fillna(medians[col], inplace=True)

    # Standardize features
    df_scaled = scaler.transform(df)

    # Make prediction
    prediction = model.predict(df_scaled)
    probability = model.predict_proba(df_scaled)[:, 1]

    # Generate risk level
    if probability[0] < 0.3:
        risk_level = "Low"
    elif probability[0] < 0.7:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    # Create feature importance interpretation
    feature_importance = {
        'Glucose': 'High glucose levels are strongly associated with diabetes risk.',
        'BMI': 'Higher BMI increases diabetes risk, especially if over 30.',
        'Age': 'Diabetes risk typically increases with age.',
        'Insulin': 'Abnormal insulin levels can indicate insulin resistance or deficiency.',
        'DiabetesPedigreeFunction': 'Family history is a significant factor in diabetes risk.',
        'BloodPressure': 'Hypertension is often comorbid with diabetes.',
        'Pregnancies': 'Multiple pregnancies can affect diabetes risk in women.',
        'SkinThickness': 'Skin thickness can be an indicator of body composition.'
    }

    # Identify key factors for this prediction
    if prediction[0] == 1:  # Diabetic
        if df['Glucose'].values[0] > 125:
            key_factors = ['High glucose level']
        else:
            key_factors = []

        if df['BMI'].values[0] > 30:
            key_factors.append('Elevated BMI')

        if df['Age'].values[0] > 50:
            key_factors.append('Age factor')

        if df['DiabetesPedigreeFunction'].values[0] > 0.8:
            key_factors.append('Strong family history')
    else:  # Non-diabetic
        key_factors = ['Normal glucose level', 'Healthy metrics']

    return {
        "prediction": int(prediction[0]),
        "probability": float(probability[0]),
        "result": "Diabetic" if prediction[0] == 1 else "Non-Diabetic",
        "confidence": f"{probability[0]*100:.2f}%" if prediction[0] == 1 else f"{(1-probability[0])*100:.2f}%",
        "risk_level": risk_level,
        "key_factors": key_factors
    }

def create_user_interface():
    \"\"\"
    Create a simple command-line interface for diabetes prediction
    \"\"\"
    print("\\n================================")
    print("  Diabetes Prediction System")
    print("================================\\n")

    try:
        pregnancies = int(input("Number of Pregnancies: "))
        glucose = float(input("Glucose Level (mg/dL): "))
        blood_pressure = float(input("Blood Pressure (mm Hg): "))
        skin_thickness = float(input("Skin Thickness (mm): "))
        insulin = float(input("Insulin Level (mu U/ml): "))
        bmi = float(input("BMI: "))
        diabetes_pedigree = float(input("Diabetes Pedigree Function: "))
        age = int(input("Age: "))

        user_data = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': diabetes_pedigree,
            'Age': age
        }

        result = predict_diabetes(user_data)

        print("\\n================================")
        print("       Prediction Results        ")
        print("================================")
        print(f"Prediction: {result['result']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Risk Level: {result['risk_level']}")

        print("================================\\n")

        return result

    except Exception as e:
        print(f"\\nError: {e}")
        print("Please enter valid values and try again.")
        return None

# Example usage
if __name__ == "__main__":
    # Example user input
    test_data = {
        'Pregnancies': 6,
        'Glucose': 148,
        'BloodPressure': 72,
        'SkinThickness': 35,
        'Insulin': 0,
        'BMI': 33.6,
        'DiabetesPedigreeFunction': 0.627,
        'Age': 50
    }

    try:
        result = predict_diabetes(test_data)
        print("\\nDiabetes Prediction Results:")
        print(f"Prediction: {result['result']}")
        print(f"Probability: {result['probability']:.4f}")
        print(f"Confidence: {result['confidence']}")


    except Exception as e:
        print(f"Error during prediction: {e}")
    """)

        print("✅ Created prediction function file: 'predict_diabetes.py'")

        return self

    def create_interactive_visualizations(self):
        """Create interactive visualizations using Plotly"""
        self.print_section_header("Interactive Visualizations")

        print("🎨 Creating interactive visualizations...")

        # 1. Interactive feature importance
        if self.feature_importance is not None:
            fig = px.bar(
                self.feature_importance,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Feature Importance for Diabetes Prediction',
                labels={'Importance': 'Importance Score', 'Feature': 'Feature'},
                color='Importance',
                color_continuous_scale='viridis'
            )

            fig.update_layout(
                height=600,
                width=900,
                template='plotly_white',
                yaxis={'categoryorder': 'total ascending'}
            )

            fig.show()

        # 2. Interactive correlation heatmap
        corr_matrix = self.df.corr()

        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect='auto',
            color_continuous_scale='RdBu_r',
            title='Interactive Correlation Matrix'
        )

        fig.update_layout(
            height=800,
            width=800,
            template='plotly_white'
        )

        fig.show()

        # 3. Interactive scatter plot matrix of key features
        top_features = self.df.corr()['Outcome'].abs().sort_values(ascending=False).index[:4].tolist()

        fig = px.scatter_matrix(
            self.df,
            dimensions=top_features,
            color='Outcome',
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
            title='Scatter Matrix of Key Features',
            labels={col: col for col in top_features}
        )

        fig.update_layout(
            height=800,
            width=900,
            template='plotly_white'
        )

        fig.show()

        # 4. Interactive distribution plots
        for feature in ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction']:
            fig = px.histogram(
                self.df,
                x=feature,
                color='Outcome',
                marginal='box',
                color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                title=f'Distribution of {feature} by Diabetes Outcome',
                labels={feature: feature, 'Outcome': 'Diabetes Outcome'},
                barmode='overlay',
                opacity=0.7
            )

            fig.update_layout(
                height=500,
                width=900,
                template='plotly_white',
                legend_title_text='Diabetes Outcome',
                xaxis_title=feature,
                yaxis_title='Count'
            )

            fig.show()

        # 5. Interactive ROC curve
        if hasattr(self, 'model') and self.model is not None:
            y_pred_proba = self.model.predict_proba(self.X_test_scaled)[:, 1]
            fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            # Create a DataFrame for the plot
            roc_df = pd.DataFrame({
                'False Positive Rate': fpr,
                'True Positive Rate': tpr,
                'Threshold': thresholds
            })

            fig = px.line(
                roc_df,
                x='False Positive Rate',
                y='True Positive Rate',
                title=f'ROC Curve (AUC = {roc_auc:.4f})',
                line_shape='spline',
                hover_data=['Threshold'],
                labels={'False Positive Rate': 'False Positive Rate', 'True Positive Rate': 'True Positive Rate'}
            )

            # Add a diagonal reference line
            fig.add_shape(
                type='line',
                line=dict(dash='dash', color='gray'),
                x0=0, y0=0, x1=1, y1=1
            )

            fig.update_layout(
                height=600,
                width=800,
                template='plotly_white',
                xaxis=dict(
                    title='False Positive Rate',
                    range=[0, 1],
                    constrain='domain'
                ),
                yaxis=dict(
                    title='True Positive Rate',
                    range=[0, 1],
                    scaleanchor="x",
                    scaleratio=1,
                    constrain='domain'
                )
            )

            fig.show()

        # 6. 3D scatter plot of top 3 features
        if len(top_features) >= 3:
            fig = px.scatter_3d(
                self.df,
                x=top_features[0],
                y=top_features[1],
                z=top_features[2],
                color='Outcome',
                color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                title=f'3D Scatter Plot of Top 3 Features',
                labels={
                    top_features[0]: top_features[0],
                    top_features[1]: top_features[1],
                    top_features[2]: top_features[2],
                    'Outcome': 'Diabetes Outcome'
                }
            )

            fig.update_layout(
                height=800,
                width=900,
                template='plotly_white'
            )

            fig.show()

        return self

    def run_analysis(self, do_hyperparameter_tuning=False):
        """Run the entire analysis pipeline"""
        self.print_section_header("Starting Diabetes Prediction Analysis")

        start_time = time.time()

        # Run all steps
        self.load_data()
        self.visualize_data_distributions()
        self.analyze_correlations()
        self.preprocess_data()
        self.feature_selection_analysis()
        self.train_model()

        if do_hyperparameter_tuning:
            self.hyperparameter_tuning(quick=True)

        self.evaluate_model()
        self.analyze_misclassifications()
        self.create_prediction_function()
        self.create_interactive_visualizations()

        total_time = time.time() - start_time

        self.print_section_header("Analysis Complete")
        print(f"✅ Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print("✅ All analysis steps completed successfully")
        print("✅ Trained model saved as 'diabetes_model.pkl'")
        print("✅ Prediction function saved as 'predict_diabetes.py'")

        if do_hyperparameter_tuning:
            print("✅ Tuned model saved as 'diabetes_model_tuned.pkl'")

        print("\n🚀 You can now use the prediction function to make diabetes predictions!")

        return self


# Run the entire analysis
if __name__ == "__main__":
    # Check if we're running in Google Colab
    try:
        import google.colab
        in_colab = True
    except ImportError:
        in_colab = False

    # Inform the user
    if in_colab:
        print("Running in Google Colab environment")
    else:
        print("Running in local environment")

    # Set plot style for better visualization in Colab
    if in_colab:
        plt.style.use('fivethirtyeight')
        sns.set_context("notebook", font_scale=1.2)

    # Create and run the analysis pipeline
    analyzer = DiabetesPredictionAnalysis("diabetes.csv")
    analyzer.run_analysis(do_hyperparameter_tuning=True)

    # Example of making a prediction
    print("\n🔮 Example Prediction:")
    try:
        #from predict_diabetes import predict_diabetes

        test_data = {
            'Pregnancies': 10,
            'Glucose': 115,
            'BloodPressure': 0,
            'SkinThickness': 0,
            'Insulin': 0,
            'BMI': 35.3,
            'DiabetesPedigreeFunction': 0.134,
            'Age': 29
        }

        result = predict_diabetes(test_data)

        print("\n========================================")
        print("        Diabetes Prediction Result")
        print("========================================")
        print(f"Prediction: {result['result']}")
        print(f"Confidence: {result['confidence']}")
        print("========================================")
    except Exception as e:
        print(f"Error making example prediction: {e}")