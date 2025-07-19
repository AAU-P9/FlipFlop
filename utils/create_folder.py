import os
import shutil

def create_and_move_folders(source_base_path, dest_base_path):
    # List of folder names
    folders = [
        "accuracy", "adam", "addBiasResidualLayerNorm", "attention", "attentionMultiHead",
        "backprop", "bincount", "bn", "channelShuffle", "channelSum", "clink",
        "concat", "crossEntropy", "dense-embedding", "dropout", "dwconv", "dwconv1d",
        "expdist", "flip", "gd", "gelu", "ge-spmm", "glu", "gmm", "gru", "kalman",
        "kmc", "kmeans", "knn", "layernorm", "lda", "lif", "logprob", "lr", "lrn",
        "mask", "matern", "maxpool3d", "mcpr", "meanshift", "mf-sgd", "mmcsf",
        "mnist", "mrc", "multinomial", "nlll", "nonzero", "overlay", "p4",
        "page-rank", "permute", "perplexity", "pointwise", "pool", "qkv",
        "qtclustering", "remap", "relu", "resnet-kernels", "rowwiseMoments",
        "rotary", "sampling", "scel", "softmax", "softmax-fused", "softmax-online",
        "stddev", "streamcluster", "swish", "unfold", "vol2col", "wedford",
        "winograd", "word2vec"
    ]

    # Create destination folders and move content
    for folder in folders:
        # Create full paths
        source_path = os.path.join(source_base_path, folder + "-cuda")
        dest_path = os.path.join(dest_base_path, folder)

        # Create destination folder
        os.makedirs(dest_path, exist_ok=True)
        print(f"Created folder: {dest_path}")

        # Move content if source folder exists
        # if os.path.exists(source_path):
        #     try:
        #         # List all items in source folder
        #         items = os.listdir(source_path)
        #         for item in items:
        #             src_item = os.path.join(source_path, item)
        #             dst_item = os.path.join(dest_path, item)
        #             shutil.move(src_item, dst_item)
        #         print(f"Moved contents from {source_path} to {dest_path}")
        #     except Exception as e:
        #         print(f"Error moving contents for {folder}: {str(e)}")
        # else:
        #     print(f"Source folder not found: {source_path}")

# Usage example
if __name__ == "__main__":
    source_base_path = "/home/anonymous/flipflop/HeCBench/src"  # Replace with source path
    dest_base_path = "/home/anonymous/flipflop/cuda_kernel_energy_empirical/rq1_plots"  # Replace with destination path
    
    create_and_move_folders(source_base_path, dest_base_path)