import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class AadharAnalyzer:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.model_results = {}

    def load_and_clean_data(self):
        """Loads CSV and performs basic cleaning."""
        print("Loading data...")
        try:
            self.df = pd.read_csv(self.filepath)
            
            # Convert date to datetime objects
            self.df['date'] = pd.to_datetime(self.df['date'], format='%d-%m-%Y', errors='coerce')
            
            # Create a Total Enrolment column
            self.df['total_enrolment'] = (
                self.df['age_0_5'] + 
                self.df['age_5_17'] + 
                self.df['age_18_greater']
            )
            
            # Drop rows with missing values
            self.df.dropna(inplace=True)
            print(f"Data loaded successfully. {len(self.df)} records found.")
            return self.df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def perform_clustering(self, n_clusters=3):
        """
        Uses K-Means to cluster data based on age demographics.
        This helps identify if a record is 'Child Focused', 'Adult Focused', etc.
        """
        print(f"\n--- Running K-Means Clustering (n={n_clusters}) ---")
        
        # Select features for clustering
        features = ['age_0_5', 'age_5_17', 'age_18_greater']
        X = self.df[features]
        
        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.df['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Analyze clusters
        cluster_summary = self.df.groupby('cluster')[features].mean()
        
        # Heuristic to name the clusters automatically
        cluster_names = {}
        for cluster_id in cluster_summary.index:
            row = cluster_summary.loc[cluster_id]
            max_cat = row.idxmax()
            cluster_names[cluster_id] = f"Dominant: {max_cat}"
            
        self.df['cluster_name'] = self.df['cluster'].map(cluster_names)
        
        print("Clustering Complete. Identified patterns:")
        print(cluster_summary)
        return cluster_summary

    def detect_anomalies(self, contamination=0.01):
        """
        Uses Isolation Forest to find unusual enrolment spikes.
        contamination: percentage of data expected to be anomalies.
        """
        print(f"\n--- Running Isolation Forest for Anomaly Detection ---")
        
        features = ['total_enrolment']
        X = self.df[features]
        
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        self.df['is_anomaly'] = iso_forest.fit_predict(X)
        
        # -1 indicates anomaly, 1 indicates normal
        anomalies = self.df[self.df['is_anomaly'] == -1]
        
        print(f"Detected {len(anomalies)} anomalies (Unusual enrolment volumes).")
        if not anomalies.empty:
            print("Top 5 Anomalous Records:")
            print(anomalies[['date', 'state', 'district', 'total_enrolment']].head(5))
        
        return anomalies

    def generate_geographic_insights(self):
        """Aggregates data by State/District to find hotspots."""
        print(f"\n--- Generating Geographic Insights ---")
        
        # Top States by Total Enrolment
        state_stats = self.df.groupby('state')['total_enrolment'].sum().sort_values(ascending=False).head(5)
        
        # Top Districts for Child Enrolment (0-5)
        child_districts = self.df.groupby(['state', 'district'])['age_0_5'].sum().sort_values(ascending=False).head(5)
        
        print("\nTop 5 States by Volume:")
        print(state_stats)
        
        print("\nTop 5 Districts for Child Enrolment (0-5 years):")
        print(child_districts)
        
        return state_stats, child_districts

    def run_full_analysis(self):
        """Runs the complete pipeline."""
        if self.load_and_clean_data() is not None:
            self.perform_clustering()
            self.detect_anomalies()
            self.generate_geographic_insights()
            print("\nAnalysis Complete.")

# --- Execution Block ---
# This block ensures the code runs only when executed directly, 
# not when imported by your Flask/web app (unless you want it to).

if __name__ == "__main__":
    # Replace with your actual file path
    FILE_PATH = 'api_data_aadhar_enrolment_0_500000.csv'
    
    analyzer = AadharAnalyzer(FILE_PATH)
    analyzer.run_full_analysis()