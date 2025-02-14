import torch

base_config = {

    "device": "cuda:0", 
    "clustering": {
        "init": "k-means++",
        "random_state": 77,
        "max_iter": 300,
        "tolerance": 1e-4
    }
}

dataset_configs = {
    "cora": {
        **base_config,
        "rho": 0.5,
        "output_file": "prototypes_cora.pt",
        "batch_size": 50,
        "clustering": {
            **base_config["clustering"],
            "max_iter": 200  
        }
    },
    
    "citeseer": {
        **base_config,
        "rho": 0.5,
        "output_file": "prototypes_citeseer.pt",
        "batch_size": 2048,
        "clustering": {
            **base_config["clustering"],
            "tolerance": 1e-3  # 适当放宽收敛阈值
        }
    },
    
    "ogbn-arxiv": {
        **base_config,
        "rho": 0.005,
        "output_file": "prototypes_arxiv.pt",
        "batch_size": 4096,
        "clustering": {
            **base_config["clustering"],
            "max_iter": 400  # 中等规模数据集增加迭代
        }
    },
    
    "reddit": {
        **base_config,
        "rho": 0.001,
        "output_file": "prototypes_reddit.pt",
        "batch_size": 8192,
        "clustering": {
            **base_config["clustering"],
            "init": "random",  # 大数据集使用随机初始化更快
            "max_iter": 150
        }
    },
    
    "flickr": {
        **base_config,
        "rho": 0.005,
        "output_file": "prototypes_flickr.pt",
        "batch_size": 16384,
        "clustering": {
            **base_config["clustering"],
            "max_iter": 200,
            "tolerance": 1e-3
        }
    }
}

def get_config(dataset_name: str):

    config = dataset_configs.get(dataset_name.lower())
    if not config:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Available: {list(dataset_configs.keys())}")

    config["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
    return config
