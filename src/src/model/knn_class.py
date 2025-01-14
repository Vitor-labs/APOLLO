

class KNNClassifier:
    def __init__(self, embeddings: np.ndarray, syndrome_ids: List):
        """
        Initialize KNN classifier with embeddings and labels
        
        Args:
            embeddings: Array of shape (n_samples, n_features)
            syndrome_ids: List of syndrome IDs for each embedding
        """
        self.embeddings = embeddings
        # Encode syndrome IDs to numerical labels
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(syndrome_ids)
        self.n_classes = len(set(self.labels))
        
        # Initialize scalers
        self.scaler = StandardScaler()
        
    def prepare_data(self) -> np.ndarray:
        """
        Scale the embeddings
        
        Returns:
            Scaled embeddings array
        """
        return self.scaler.fit_transform(self.embeddings)
    
    def evaluate_knn(self, X: np.ndarray, y: np.ndarray, k: int, 
                    metric: str, n_folds: int = 10) -> Dict:
        """
        Evaluate KNN model using cross-validation
        
        Args:
            X: Scaled embeddings
            y: Labels
            k: Number of neighbors
            metric: Distance metric ('cosine' or 'euclidean')
            n_folds: Number of cross-validation folds
            
        Returns:
            Dictionary containing evaluation metrics
        """
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Initialize metrics storage
        aucs = []
        f1_scores = []
        top1_accuracies = []
        top3_accuracies = []
        
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train KNN model
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
            knn.fit(X_train, y_train)
            
            # Get probabilities for each class
            proba = knn.predict_proba(X_val)
            
            # Calculate metrics
            # AUC (one-vs-rest)
            auc = roc_auc_score(y_val, proba, multi_class='ovr')
            aucs.append(auc)
            
            # F1 score (weighted average)
            y_pred = knn.predict(X_val)
            f1 = f1_score(y_val, y_pred, average='weighted')
            f1_scores.append(f1)
            
            # Top-k accuracy
            top1_acc = accuracy_score(y_val, y_pred)
            top1_accuracies.append(top1_acc)
            
            # Top-3 accuracy
            top3_pred = np.argsort(proba, axis=1)[:, -3:]
            top3_acc = np.mean([1 if y_val[i] in top3_pred[i] else 0 
                              for i in range(len(y_val))])
            top3_accuracies.append(top3_acc)
        
        return {
            'auc_mean': np.mean(aucs),
            'auc_std': np.std(aucs),
            'f1_mean': np.mean(f1_scores),
            'f1_std': np.std(f1_scores),
            'top1_acc_mean': np.mean(top1_accuracies),
            'top1_acc_std': np.std(top1_accuracies),
            'top3_acc_mean': np.mean(top3_accuracies),
            'top3_acc_std': np.std(top3_accuracies)
        }
    
    def bayesian_optimize(self, X: np.ndarray, y: np.ndarray, 
                         metric: str, n_folds: int = 10) -> Dict:
        """
        Perform Bayesian optimization to find optimal k
        
        Args:
            X: Scaled embeddings
            y: Labels
            metric: Distance metric ('cosine' or 'euclidean')
            n_folds: Number of cross-validation folds
            
        Returns:
            Dictionary containing optimization results
        """
        def objective(k):
            k = int(k)  # Bayesian optimization works with floats
            metrics = self.evaluate_knn(X, y, k, metric, n_folds)
            return metrics['f1_mean']  # Optimize for F1 score
        
        optimizer = BayesianOptimization(
            f=objective,
            pbounds={'k': (1, 15)},
            random_state=42
        )
        
        optimizer.maximize(init_points=5, n_iter=10)
        
        return {
            'optimal_k': int(optimizer.max['params']['k']),
            'best_score': optimizer.max['target']
        }
    
    def compare_metrics(self, k_range: range = range(1, 16)) -> Dict:
        """
        Compare performance between cosine and euclidean metrics
        
        Args:
            k_range: Range of k values to test
            
        Returns:
            Dictionary containing comparison results
        """
        X = self.prepare_data()
        results = {
            'cosine': {},
            'euclidean': {}
        }
        
        for metric in ['cosine', 'euclidean']:
            # First perform Bayesian optimization
            opt_results = self.bayesian_optimize(X, self.labels, metric)
            optimal_k = opt_results['optimal_k']
            
            # Then evaluate all k values for comparison
            k_results = {}
            for k in k_range:
                metrics = self.evaluate_knn(X, self.labels, k, metric)
                k_results[k] = metrics
            
            results[metric] = {
                'optimal_k': optimal_k,
                'best_score': opt_results['best_score'],
                'all_k_results': k_results
            }
        
        return results

    def plot_comparison(self, results: Dict) -> None:
        """
        Plot comparison of metrics between distance metrics
        
        Args:
            results: Results dictionary from compare_metrics
        """
        metrics = ['f1_mean', 'auc_mean', 'top1_acc_mean', 'top3_acc_mean']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for idx, metric in enumerate(metrics):
            for dist_metric in ['cosine', 'euclidean']:
                k_values = list(results[dist_metric]['all_k_results'].keys())
                metric_values = [results[dist_metric]['all_k_results'][k][metric] 
                               for k in k_values]
                
                axes[idx].plot(k_values, metric_values, 
                             label=f'{dist_metric}', marker='o')
                
            axes[idx].set_xlabel('k')
            axes[idx].set_ylabel(metric)
            axes[idx].set_title(f'{metric} vs k')
            axes[idx].legend()
            axes[idx].grid(True)
        
        plt.tight_layout()
        plt.show()

def main():
    # Example usage
    # Load your data first (using previous artifact's functions)
    file_path = 'mini_gm_public_v0.1.p'
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    embeddings, syndrome_ids, _, _ = flatten_hierarchical_data(data)
    
    # Create classifier
    classifier = KNNClassifier(embeddings, syndrome_ids)
    
    # Compare metrics and optimize
    results = classifier.compare_metrics()
    
    # Print results
    print("\nOptimization Results:")
    for metric in ['cosine', 'euclidean']:
        print(f"\n{metric.capitalize()} Distance:")
        print(f"Optimal k: {results[metric]['optimal_k']}")
        print(f"Best F1 Score: {results[metric]['best_score']:.4f}")
        
        # Print detailed metrics for optimal k
        opt_k = results[metric]['optimal_k']
        opt_metrics = results[metric]['all_k_results'][opt_k]
        print(f"\nDetailed metrics at k={opt_k}:")
        print(f"AUC: {opt_metrics['auc_mean']:.4f} (±{opt_metrics['auc_std']:.4f})")
        print(f"F1: {opt_metrics['f1_mean']:.4f} (±{opt_metrics['f1_std']:.4f})")
        print(f"Top-1 Accuracy: {opt_metrics['top1_acc_mean']:.4f} (±{opt_metrics['top1_acc_std']:.4f})")
        print(f"Top-3 Accuracy: {opt_metrics['top3_acc_mean']:.4f} (±{opt_metrics['top3_acc_std']:.4f})")
    
    # Plot comparison
    classifier.plot_comparison(results)

if __name__ == "__main__":
    main()