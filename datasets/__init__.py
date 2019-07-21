from .product import ProductDataset

datasets = {
    'product': ProductDataset
}


def get_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)
