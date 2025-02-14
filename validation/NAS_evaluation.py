import csv
import os
import pickle as pkl
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from scipy.stats import pearsonr
from tqdm import tqdm
from torch_geometric.data import Data
from models.gnn_modules import *

def save_csv(file_path: Path, data: List[float]) -> None:
    """Save numerical data to CSV file"""
    with file_path.open(mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)
    print(f"Saved CSV to {file_path}")

def load_csv(file_path: Path) -> List[float]:
    """Load numerical data from CSV file"""
    with file_path.open(mode='r') as f:
        reader = csv.reader(f)
        return [float(x) for row in reader for x in row]

def save_pkl(file_path: Path, data: object) -> None:
    """Serialize data using pickle"""
    with file_path.open('wb') as f:
        pkl.dump(data, f)
    print(f"Saved pickle to {file_path}")

def load_pkl(file_path: Path) -> object:
    """Deserialize data from pickle file"""
    with file_path.open('rb') as f:
        return pkl.load(f)

class NASValidator:
    """
    Neural Architecture Search Validator for Graph Neural Networks
    
    Parameters
    ----------
    param_space : Dict[str, List]
        Dictionary defining hyperparameter search space
    save_dir : Path
        Directory to save validation results
    device : torch.device
        Computation device (CPU/GPU)
    """
    
    def __init__(self, 
                 param_space: Dict[str, List],
                 save_dir: Path,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.param_space = param_space
        self.save_dir = save_dir
        self.device = device
        
        self.best_params = {'original': None, 'synthetic': None}
        self.results = {'original': [], 'synthetic': []}
        
        # Create parameter combinations
        self.param_combos = list(product(*param_space.values()))
        self.param_names = list(param_space.keys())
        
        # Ensure output directory exists
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _init_model(self, params: Tuple) -> torch.nn.Module:
        """Initialize APPNP model with given parameters"""
        # Unpack parameters (example format)
        n_layers, hidden_dim, alpha, activation = params
        
        return APPNP(
            in_channels=self.data.num_features,
            hidden_channels=hidden_dim,
            out_channels=self.data.num_classes,
            K=n_layers,
            alpha=alpha,
            dropout=0.5
        ).to(self.device)

    def evaluate(self, 
                 data: Data,
                 dataset_type: str = 'original',
                 num_epochs: int = 600) -> None:
        """
        Evaluate hyperparameter combinations on specified dataset
        
        Parameters
        ----------
        data : Data
            PyG Data object containing full graph information
        dataset_type : str ('original'|'synthetic')
            Type of dataset being evaluated
        num_epochs : int
            Number of training epochs
        """
        best_acc = 0.0
        
        for params in tqdm(self.param_combos, desc=f"Evaluating {dataset_type}"):
            # Initialize model
            model = self._init_model(params)
            
            # Train and evaluate
            accuracy = train_and_evaluate(
                model=model,
                data=data,
                train_mask=data.train_mask,
                test_mask=data.val_mask,  # Use validation mask for NAS
                epochs=num_epochs,
                learning_rate=0.01
            )
            
            # Record results
            self.results[dataset_type].append(accuracy)
            
            # Update best parameters
            if accuracy > best_acc:
                best_acc = accuracy
                self.best_params[dataset_type] = dict(zip(self.param_names, params))
        
        # Save results
        self._save_results(dataset_type)

    def _save_results(self, dataset_type: str) -> None:
        """Save results for specified dataset type"""
        suffix = '' if dataset_type == 'original' else '_syn'
        
        # Save numerical results
        csv_path = self.save_dir / f'results{suffix}.csv'
        save_csv(csv_path, self.results[dataset_type])
        
        # Save best parameters
        pkl_path = self.save_dir / f'best_params{suffix}.pkl'
        save_pkl(pkl_path, self.best_params[dataset_type])

    def validate_best(self, 
                      data: Data,
                      dataset_type: str = 'original') -> float:
        """
        Validate best parameters on test set
        
        Parameters
        ----------
        data : Data
            PyG Data object containing test mask
        dataset_type : str
            Type of dataset to validate
            
        Returns
        -------
        test_accuracy : float
            Accuracy on test set
        """
        # Load best parameters if not available
        if not self.best_params[dataset_type]:
            suffix = '' if dataset_type == 'original' else '_syn'
            self.best_params[dataset_type] = load_pkl(self.save_dir / f'best_params{suffix}.pkl')
        
        # Initialize best model
        best_params = list(self.best_params[dataset_type].values())
        model = self._init_model(best_params)
        
        # Final evaluation on test set
        return train_and_evaluate(
            model=model,
            data=data,
            train_mask=data.train_mask,
            test_mask=data.test_mask,
            epochs=600,
            learning_rate=0.01
        )

    def calculate_correlations(self) -> Tuple[float, float]:
        """
        Calculate Pearson correlations between original and synthetic results
        
        Returns
        -------
        (acc_corr, rank_corr) : Tuple[float, float]
            Pearson correlations for accuracy values and parameter rankings
        """
        # Load results if needed
        if not self.results['original']:
            self.results['original'] = load_csv(self.save_dir / 'results.csv')
        if not self.results['synthetic']:
            self.results['synthetic'] = load_csv(self.save_dir / 'results_syn.csv')
        
        # Calculate accuracy correlation
        acc_corr, _ = pearsonr(self.results['original'], self.results['synthetic'])
        
        # Calculate rank correlation
        rank_orig = self._calculate_ranks(self.results['original'])
        rank_syn = self._calculate_ranks(self.results['synthetic'])
        rank_corr, _ = pearsonr(rank_orig, rank_syn)
        
        return acc_corr, rank_corr

    def _calculate_ranks(self, results: List[float]) -> List[int]:
        """Generate ranking for results"""
        sorted_indices = sorted(range(len(results)), key=lambda i: results[i], reverse=True)
        ranks = [0] * len(results)
        current_rank = 1
        
        for idx in sorted_indices:
            if ranks[idx] == 0:
                ranks[idx] = current_rank
                current_rank += 1
                
        return ranks