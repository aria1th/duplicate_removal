import fiftyone as fo
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import fiftyone.zoo as foz
from tqdm import tqdm
from PIL import Image
import shutil
from fiftyone import ViewField as F


def process_jpg_dir(result_dir):
    # Path to your images directory
    dataset_dir = result_dir
    # Load or create a dataset
    dataset = fo.Dataset.from_dir(dataset_dir, dataset_type=fo.types.ImageDirectory)
    print(dataset)
    #@title 3.3. Calculate Similarity (thanks ChatGPT for the batching)
    #Too lazy to make this thing async lol
    # Set the batch size
    batch_size = 1000
    #@title 3.2. Create Embeddings
    model_name = "clip-vit-base32-torch" #@param ["clip-vit-base32-torch", "mobilenet-v2-imagenet-torch"] {type: "string", allow-input: true}
    model = foz.load_zoo_model(model_name)
    embeddings = dataset.compute_embeddings(model, batch_size=250)
    similarity_matrices = []
    #@title 3.3. Calculate Similarity (thanks ChatGPT for the batching)
    #Too lazy to make this thing async lo

    batch_embeddings = np.array_split(embeddings, batch_size)
    similarity_matrices = []
    # Find the maximum size of the arrays
    max_size_x = max(array.shape[0] for array in batch_embeddings)
    max_size_y = max(array.shape[1] for array in batch_embeddings)
    batch_embeddings = [b for b in batch_embeddings if b.shape[0] > 0 and b.shape[1] > 0]


    for batch_embedding in batch_embeddings:
        similarity = cosine_similarity(batch_embedding)
        #Pad 0 for np.concatenate
        padded_array = np.zeros((max_size_x, max_size_y))
        padded_array[0:similarity.shape[0], 0:similarity.shape[1]] = similarity
        similarity_matrices.append(padded_array)

    # Concatenate the padded arrays
    similarity_matrix = np.concatenate(similarity_matrices, axis=0)
    similarity_matrix = similarity_matrix[0:embeddings.shape[0], 0:embeddings.shape[0]]

    similarity_matrix = cosine_similarity(embeddings)
    similarity_matrix -= np.identity(len(similarity_matrix))

    #@title 3.4. Generate list of images to remove and Calculate remove percentage

    threshold = 0.95 #@param {type: "number"}
    #@markdown * Any duplicate image which has similarity score > this number shall be removed
    #@markdown * If you want to change the threshold then just check the Remove percentage and decide how many images should be removed

    dataset.match(F("max_similarity") > threshold)

    id_map = [s.id for s in dataset.select_fields(["id"])]
    samples_to_remove = set()
    samples_to_keep = set()

    for idx, sample in enumerate(dataset):
        if sample.id not in samples_to_remove:
            # Keep the first instance of two duplicates
            samples_to_keep.add(sample.id)
            
            dup_idxs = np.where(similarity_matrix[idx] > threshold)[0]
            for dup in dup_idxs:
                # We kept the first instance so remove all other duplicates
                samples_to_remove.add(id_map[dup])

            if len(dup_idxs) > 0:
                sample.tags.append("has_duplicates")
                sample.save()

        else:
            sample.tags.append("duplicate")
            sample.save()

    print(f"Remove percentage: {len(samples_to_remove) / (len(samples_to_remove) + len(samples_to_keep)) * 100}")

    # reflect the changes in the dataset
    dirs_to_remove = []
    for idx, sample in enumerate(dataset):
        if sample.id in samples_to_remove:
            dirs_to_remove.append(sample.filepath)
    return dirs_to_remove

def process_dir(directory):
    """
    Converts webp -> jpg then removes duplicates, returns original file paths of removed files
    """
    path_pairs = {}
    result_dir = directory + '_jpg'
    os.makedirs(result_dir, exist_ok=True)
    for root, dirs, files in os.walk(directory):
        for file in tqdm(files, desc='Converting webp to jpg'):
            if file.endswith('.webp'):
                img = Image.open(os.path.join(root, file))
                img.save(os.path.join(result_dir, file.replace('.webp', '.jpg')))
                path_pairs[file.replace('.webp', '.jpg')] = os.path.join(root, file)
            else:
                img = None
                try:
                    img = Image.open(os.path.join(root, file))
                    img.load()
                    img.close()
                except:
                    if img is not None:
                        img.close()
                    print(f"Skipping {file}")
                    continue
                shutil.copy(os.path.join(root, file), os.path.join(result_dir, file))
                path_pairs[file] = os.path.join(root, file)
    dataset_dir = result_dir
    dirs_to_remove = process_jpg_dir(dataset_dir)
    # get original file paths
    original_files = []
    for file in dirs_to_remove:
        original_files.append(path_pairs[os.path.basename(file)])
    # remove directories
    shutil.rmtree(dataset_dir)
    return original_files

def process_subdirs(root):
    all_results = []
    dirs = os.listdir(root)
    dirs = [os.path.join(root, d) for d in dirs if os.path.isdir(os.path.join(root, d))]
    for d in tqdm(dirs, desc='Processing subdirectories'):
        all_results += process_dir(d)
    return all_results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Remove duplicates from a directory of images')
    parser.add_argument('--root', type=str, help='Directory containing subdirectories of images')
    parser.add_argument('--output', type=str, help='Output file containing paths of removed images')
    parser.add_argument('--root_only', action='store_true', help='Only process images in the root directory')
    parser.add_argument('--reflect', action='store_true', help='Reflect changes in the dataset')
    args = parser.parse_args()
    if args.root_only:
        all_results = process_dir(args.root)
    else:
        all_results = process_subdirs(args.root)
    with open(args.output, 'w', encoding='utf-8') as f:
        for item in all_results:
            f.write("%s\n" % item)
    print(f'All removed files have been written to {args.output}')
    if args.reflect:
        # read and remove files
        with open(args.output, 'r', encoding='utf-8') as f:
            removed_files = f.readlines()
            removed_files = [f.strip() for f in removed_files]
        for f in removed_files:
            os.remove(f)
