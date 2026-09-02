"""
Feature extraction and classifier training for AdaConRed.

Encodes OSCC oral-lesion or ISIC skin-lesion images with a frozen vision
foundation model (DermFoundation or MedSigLIP), trains a lightweight MLP
head on the resulting embeddings, and returns softmax outputs for the
calibration and test splits. Those outputs are the input to the adaptive
conformal redistribution stage.

Usage
-----
Set DATASET, ENCODER, and the three path constants below, then:

    from encoder_classifier import get_softmax
    softmax_calib, y_calib, softmax_test, y_test = get_softmax()

Requires HF_TOKEN in the environment when ENCODER == "MSL".
"""

import os
import random
import shutil
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from huggingface_hub import login
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# -----------------------------CONFIGURATION-----------------------------

MODEL_FILES_PATH = ""  # Source directory for DermFoundation model files
MODEL_DEST_PATH = "path_to_dir_for_model_storage"
DATA_SRC_DIR = "path_to_root_dir_of_data"

# Directory to write the .npy files 
DATA_ROOT = "path_to_dir_of_softmax_outputs"

DATASET = "OSCC"  # Options: "OSCC" or "ISIC"
AUGMENTED = False  # If True, add synthetic OCA images to the training split
SYNTHETIC_OCA_DIR = "path_to_dir_of_750_synthetic_OCA_images"
N_OCA_TRAIN_TO_TEST = 50  # Real OCA instances moved from train to test post-augmentation

ENCODER = "DF"  # Options: "DF" (DermFoundation) or "MSL" (MedSigLIP)
HF_TOKEN_ENV_VAR = "HF_TOKEN"
if DATASET == "OSCC":
    LABELS = ["Healthy", "Benign", "OPMD", "OCA"]
    patient_ids_all = []
else:
    LABELS = ["AK", "BCC", "BKL", "MEL", "NV", "SCC", "UNK"]

# -----------------------------REPRODUCIBILITY-----------------------------

def seed_everything(seed: int = 42) -> None:
    """Set all relevant random seeds for reproducibility when using a GPU."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_everything()

# --------DERMFOUNDATION ENCODER  (TensorFlow / BiT ResNet-101x3)---------

def setup_dermfoundation_encoder(
    source_path: str = MODEL_FILES_PATH,
    dest_path: str = MODEL_DEST_PATH,
):
    """
    Copy DermFoundation SavedModel files into the expected directory layout
    and load the model onto CPU.

    Expected source files
    ---------------------
    saved_model.pb
    variables.data-00000-of-00001
    variables.index
    """
    print("=== Setting up DermFoundation encoder ===")
    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)
    os.makedirs(dest_path)

    shutil.copy2(
        os.path.join(source_path, "saved_model.pb"),
        os.path.join(dest_path, "saved_model.pb"),
    )

    variables_dir = os.path.join(dest_path, "variables")
    os.makedirs(variables_dir, exist_ok=True)
    for fname in ("variables.data-00000-of-00001", "variables.index"):
        shutil.copy2(
            os.path.join(source_path, fname),
            os.path.join(variables_dir, fname),
        )

    print("Loading SavedModel onto CPU …")
    with tf.device("/cpu:0"):
        loaded_model = tf.saved_model.load(dest_path)
    print("DermFoundation encoder loaded.\n")
    return loaded_model


class DermFoundationEncoder(tf.keras.Model):
    """
    Keras wrapper around the DermFoundation BiT ResNet-101x3 SavedModel.

    Input : float32 tensor of shape (B, H, W, 3), values in [0, 1].
    Output: embedding tensor of shape (B, D).
    """

    def __init__(self, loaded_model, **kwargs):
        super().__init__(**kwargs)
        self.loaded_model = loaded_model
        self.infer_fn = loaded_model.signatures["serving_default"]

    def call(self, inputs, training=False):
        def _encode_batch(images):
            """Serialize a NumPy batch as TFRecord examples expected by the SavedModel."""
            serialized = []
            for img in images.numpy():
                img_uint8 = (img * 255.0).astype(np.uint8)
                buf = BytesIO()
                Image.fromarray(img_uint8).save(buf, "PNG")
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "image/encoded": tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[buf.getvalue()])
                            )
                        }
                    )
                )
                serialized.append(example.SerializeToString())
            return np.array(serialized, dtype=object)

        serialized_inputs = tf.py_function(_encode_batch, [inputs], tf.string)
        serialized_inputs = tf.ensure_shape(serialized_inputs, [None])

        with tf.device("/cpu:0"):
            output = self.infer_fn(inputs=serialized_inputs)

        return output["embedding"]


def preprocess_image_dermfoundation(
    img_pil: Image.Image, target_size: int = 448
) -> np.ndarray:
    """Resize and normalize a PIL image to float32 [0, 1] for DermFoundation."""
    img = img_pil.resize((target_size, target_size), Image.Resampling.LANCZOS)
    return np.array(img).astype(np.float32) / 255.0


# -------MEDSIGLIP ENCODER  (PyTorch / google/medsiglip-448)--------

MEDSIGLIP_MODEL_ID = "google/medsiglip-448"


def _authenticate_huggingface() -> None:
    """
    Reads the Hugging Face token from the environment and logs in.
    Raises RuntimeError if the variable is unset or empty.
    """
    hf_token = os.environ.get(HF_TOKEN_ENV_VAR, "").strip()
    if not hf_token:
        raise RuntimeError(
            f"Environment variable '{HF_TOKEN_ENV_VAR}' is not set or is empty. "
            "Export your Hugging Face token before running."
        )
    login(token=hf_token)
    print("Hugging Face authentication successful.")


def setup_medsiglip_encoder():
    """
    Authenticate with Hugging Face, then download (or load from cache) the
    MedSigLIP processor and vision model, placing the model on GPU if available.

    Returns
    -------
    processor : AutoProcessor
    model     : AutoModelForZeroShotImageClassification  (eval mode)
    device    : torch.device
    """
    print(f"=== Setting up MedSigLIP encoder ({MEDSIGLIP_MODEL_ID}) ===")
    _authenticate_huggingface()

    processor = AutoProcessor.from_pretrained(MEDSIGLIP_MODEL_ID)
    model = AutoModelForZeroShotImageClassification.from_pretrained(MEDSIGLIP_MODEL_ID)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print(f"MedSigLIP encoder loaded on {device}.\n")
    return processor, model, device


def extract_image_features_medsiglip(
    batch_images: list,
    processor,
    model,
    device: torch.device,
) -> np.ndarray:
    """
    Run a list of PIL images through MedSigLIP and return pooled embeddings
    as a NumPy array of shape (B, D).
    """
    inputs = processor(images=batch_images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        features = model.get_image_features(pixel_values=pixel_values)

    pooled = features.pooler_output if hasattr(features, "pooler_output") else features
    return pooled.cpu().numpy()


# =============================================================================
# EMBEDDING EXTRACTION  (encoder-agnostic)
# =============================================================================


def extract_and_save_embeddings(tasks: list, batch_size: int = 16):
    """
    Iterate over a list of directory tasks, encode every image with the
    encoder selected via the global ENCODER flag, and accumulate embeddings
    and labels in memory.

    Parameters
    ----------
    tasks : list of dicts with keys ``path`` and ``label``.
    batch_size : images processed per forward pass.

    Returns
    -------
    final_embeddings_array : np.ndarray, shape (N, D)
    all_labels             : list of str, length N
    """
    # --- Initialise the selected encoder ---
    if ENCODER == "DF":
        loaded_model = setup_dermfoundation_encoder()
        df_encoder = DermFoundationEncoder(loaded_model)
        msl_processor = msl_model = msl_device = None
    elif ENCODER == "MSL":
        msl_processor, msl_model, msl_device = setup_medsiglip_encoder()
        df_encoder = None
    else:
        raise ValueError(f"Unknown ENCODER value: '{ENCODER}'. Choose 'DF' or 'MSL'.")

    all_embeddings: list = []
    all_labels: list = []

    for task in tasks:
        folder_path = task["path"]
        label_name = task["label"]

        if not os.path.exists(folder_path):
            print(f"Directory not found, skipping: {folder_path}")
            continue

        valid_extensions = (".png", ".jpg", ".jpeg")
        image_files = sorted(
            f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)
        )
        if not image_files:
            print(f"No images found in {folder_path}, skipping.")
            continue

        print(
            f"\nProcessing {len(image_files)} images from '{folder_path}' as label '{label_name}'"
        )

        current_batch_imgs: list = []  # preprocessed images for DF
        current_batch_pils: list = []  # raw PIL images for MSL
        current_batch_count: int = 0

        def _flush_batch():
            """Forward the accumulated batch through the active encoder."""
            if current_batch_count == 0:
                return

            if ENCODER == "DF":
                tensor_in = tf.constant(np.array(current_batch_imgs), dtype=tf.float32)
                batch_embs = df_encoder(tensor_in, training=False).numpy()
                current_batch_imgs.clear()
            else:
                batch_embs = extract_image_features_medsiglip(
                    current_batch_pils, msl_processor, msl_model, msl_device
                )
                current_batch_pils.clear()

            all_embeddings.append(batch_embs)
            all_labels.extend([label_name] * batch_embs.shape[0])

        for img_name in tqdm(image_files):
            img_path = os.path.join(folder_path, img_name)
            try:
                # OSCC-specific: skip a known corrupt image
                # (truncated JPEG that raises on decode)
                if DATASET == "OSCC" and img_name == "N-260-01.jpg":
                    continue

                img_pil = Image.open(img_path).convert("RGB")

                # Track patient IDs only after a successful decode, so that
                # patient_ids_all stays index-aligned with the embeddings.
                if DATASET == "OSCC":
                    parts = os.path.splitext(img_name)[0].split("-")
                    patient_ids_all.append(f"{parts[0]}-{parts[1]}")

                if ENCODER == "DF":
                    current_batch_imgs.append(preprocess_image_dermfoundation(img_pil))
                else:
                    current_batch_pils.append(img_pil)

                current_batch_count += 1

                if current_batch_count >= batch_size:
                    _flush_batch()
                    current_batch_count = 0

            except Exception as exc:
                print(f"  Error processing {img_path}: {exc}")

        # Flush the final (possibly incomplete) batch
        _flush_batch()

    final_embeddings_array = np.vstack(all_embeddings)
    return final_embeddings_array, all_labels


# =============================================================================
# EXTRACTION TRIGGER
# =============================================================================


def run_extraction(
    base_dir: str = DATA_SRC_DIR,
    labels_to_process=LABELS,
):
    """
    Build per-class directory tasks from ``base_dir`` and run embedding
    extraction.
    """
    global patient_ids_all
    patient_ids_all = []  # reset so repeated calls don't accumulate stale IDs
    
    processing_tasks = [
        {"path": os.path.join(base_dir, label), "label": label}
        for label in labels_to_process
    ]
    return extract_and_save_embeddings(
        tasks=processing_tasks,
        batch_size=64,  # Adjust according to available VRAM
    )


# =============================================================================
# UTILITIES
# =============================================================================


def print_dist(name: str, y_arr: np.ndarray) -> None:
    """Print the per-class sample counts for a given split."""
    unique, counts = np.unique(y_arr, return_counts=True)
    dist = dict(zip(unique, counts))
    dist_str = " | ".join(f"{LABELS[k]}: {dist.get(k, 0)}" for k in range(len(LABELS)))
    print(f"{name:12} | Total: {len(y_arr):5} | {dist_str}")


def create_dataloader(
    X,
    y,
    batch_size: int = 32,
    shuffle: bool = False,
) -> DataLoader:
    """Wrap NumPy arrays in a PyTorch DataLoader."""
    tensor_x = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.long)
    return DataLoader(
        TensorDataset(tensor_x, tensor_y), batch_size=batch_size, shuffle=shuffle
    )


# -----------------------------MLP CLASSIFIERS-----------------------------

class MLPClassifier_ISIC(nn.Module):
    """
    MLP classification head for the 7-class ISIC skin-lesion benchmark.

    Architecture: input_dim -> 128 (ReLU) -> 32 (ReLU) -> num_classes.
    A second hidden layer is used here to accommodate the larger label
    space, as described in Section 3.4 of the paper.
    """

    def __init__(self, input_dim: int, num_classes: int = 7):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class MLPClassifier_OSCC(nn.Module):
    """
    MLP classification head for the 4-class OSCC oral-lesion benchmark.

    Architecture: input_dim -> 128 (ReLU) -> num_classes.
    A single hidden layer, as described in Section 3.4 of the paper.
    """

    def __init__(self, input_dim: int, num_classes: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


# -----------------------------SOFTMAX OUTPUT COLLECTION-----------------------------

def save_prediction_softmax(loader: DataLoader, model: nn.Module):
    """
    Run ``model`` over ``loader`` in eval mode and collect per-sample
    softmax probabilities and ground-truth labels.

    Returns
    -------
    all_probs : np.ndarray, shape (N, C)
    all_true  : np.ndarray, shape (N,)
    """
    model.eval()
    all_probs, all_true = [], []

    with torch.no_grad():
        for inputs, targets in loader:
            probs = torch.softmax(model(inputs), dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_true.extend(targets.cpu().numpy())

    return np.array(all_probs), np.array(all_true)


# -----------------------------MAIN PIPELINE-----------------------------

def get_softmax():
    """
    End-to-end pipeline:
      1. Extract embeddings via the selected encoder.
      2. Perform dataset-appropriate patient-level / stratified splits.
      3. Train a lightweight MLP classifier on the training split.
      4. Return softmax outputs for the calibration and test splits.

    Returns
    -------
    softmax_calib : np.ndarray, shape (N_calib, C)
    y_calib       : np.ndarray, shape (N_calib,)
    softmax_test  : np.ndarray, shape (N_test, C)
    y_test        : np.ndarray, shape (N_test,)
    """
    global patient_ids_all

    # --- Embedding extraction ---
    embs, labs = run_extraction()

    emb_to_lab: dict = {}
    for emb, lab in zip(embs, labs):
        emb_to_lab.setdefault(lab, []).append(emb)

    class_to_idx = {c: i for i, c in enumerate(LABELS)}
    X_list, y_list = [], []

    print("Loading embeddings …")
    for cls in LABELS:
        data = np.array(emb_to_lab.get(cls, []))
        X_list.append(data)
        y_list.extend([class_to_idx[cls]] * data.shape[0])

    X_all = np.vstack(X_list)
    y_all = np.array(y_list)
    print(f"Total embeddings: {X_all.shape[0]}  |  feature dim: {X_all.shape[1]}\n")

    # --- Data splitting ---
    if DATASET == "ISIC":
        # Three-way stratified split: train / validation / calibration / test
        TEST_FRAC = 12 / 19
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X_all, y_all, test_size=0.20, stratify=y_all, random_state=42
        )
        X_rem, X_test, y_rem, y_test = train_test_split(
            X_holdout,
            y_holdout,
            test_size=TEST_FRAC,
            stratify=y_holdout,
            random_state=42,
        )
        X_val, X_calib, y_val, y_calib = train_test_split(
            X_rem, y_rem, test_size=(6 / 7), stratify=y_rem, random_state=42
        )

    elif DATASET == "OSCC":
        # Patient-level group splits to prevent data leakage across clinical images
        total_samples = len(X_all)
        patient_ids_all = np.array(patient_ids_all)

        # 4-class setting (Healthy, Benign, OPMD, OCA), non-augmented.
        # Test: 198 benign + 368 OPMD + 34 OCA + 131 healthy = 731.
        target_test, target_calib, target_val = 731, 300, 50
        holdout_fraction = (target_test + target_calib + target_val) / total_samples
        test_fraction_of_holdout = target_test / (
            target_test + target_calib + target_val
        )
        calib_fraction_of_rem = target_calib / (target_calib + target_val)

        gss1 = GroupShuffleSplit(
            n_splits=1, test_size=holdout_fraction, random_state=42
        )
        train_idx, holdout_idx = next(gss1.split(X_all, y_all, patient_ids_all))
        
        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_holdout, y_holdout = X_all[holdout_idx], y_all[holdout_idx]
        groups_holdout = patient_ids_all[holdout_idx]

        gss2 = GroupShuffleSplit(
            n_splits=1, test_size=test_fraction_of_holdout, random_state=42
        )
        rem_idx, test_idx = next(gss2.split(X_holdout, y_holdout, groups_holdout))
        X_rem, y_rem = X_holdout[rem_idx], y_holdout[rem_idx]
        X_test, y_test = X_holdout[test_idx], y_holdout[test_idx]
        groups_rem = groups_holdout[rem_idx]

        gss3 = GroupShuffleSplit(
            n_splits=1, test_size=calib_fraction_of_rem, random_state=42
        )
        val_idx, calib_idx = next(gss3.split(X_rem, y_rem, groups_rem))
        X_val, y_val = X_rem[val_idx], y_rem[val_idx]
        X_calib, y_calib = X_rem[calib_idx], y_rem[calib_idx]
        
        # GroupShuffleSplit's test_size is a fraction of GROUPS, not samples, so
        # the realised split sizes differ from the targets above. The seed below
        # was selected to bring them within tolerance; report the realised sizes.
        print(
            f"Realised split sizes -> train: {len(y_train)}, val: {len(y_val)}, "
            f"calib: {len(y_calib)}, test: {len(y_test)}"
        )
        assert len(y_test) == target_test, (
            f"Test split is {len(y_test)}, expected {target_test}. "
            "Search random_state values until the realised size matches."
        )

    print("=" * 90)
    print(f"CLASS DISTRIBUTIONS  ({DATASET})")
    print("=" * 90)
    for split_name, y_split in [
        ("Train", y_train),
        ("Validation", y_val),
        ("Calibration", y_calib),
        ("Test", y_test),
    ]:
        print_dist(split_name, y_split)
    print("=" * 90 + "\n")

    # --- DataLoader creation ---
    train_loader = create_dataloader(X_train, y_train, shuffle=True)
    val_loader = create_dataloader(X_val, y_val, shuffle=False)
    calib_loader = create_dataloader(X_calib, y_calib, shuffle=False)
    test_loader = create_dataloader(X_test, y_test, shuffle=False)

    # --- Model, loss, and optimiser ---
    input_dim = X_all.shape[1]
    if DATASET == "ISIC":
        model = MLPClassifier_ISIC(input_dim=input_dim, num_classes=len(LABELS))
    else:
        model = MLPClassifier_OSCC(input_dim=input_dim)

    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # --- Training loop ---
    epochs = 500
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_state = -1.0, None
    print("Starting training …")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss, all_preds, all_targets = 0.0, [], []
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item() * inputs.size(0)
                all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = accuracy_score(all_targets, all_preds)
        
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:03d}/{epochs}  "
                f"Train Loss: {epoch_train_loss:.4f}  "
                f"Val Loss: {epoch_val_loss:.4f}  "
                f"Val Acc: {epoch_val_acc:.4f}"
            )

    # --- Restore the best validation checkpoint before inference ---
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best checkpoint (val acc: {best_val_acc:.4f})")

    # --- Collect softmax outputs for downstream conformal prediction ---
    softmax_calib, y_calib_out = save_prediction_softmax(calib_loader, model)
    
    softmax_test, y_test_out = save_prediction_softmax(test_loader, model)
    
    np.save(os.path.join(DATA_ROOT, "calib_softmax_scores.npy"), softmax_calib)
    np.save(os.path.join(DATA_ROOT, "y_calib.npy"), y_calib_out)
    np.save(os.path.join(DATA_ROOT, "test_softmax_scores.npy"), softmax_test)
    np.save(os.path.join(DATA_ROOT, "y_test.npy"), y_test_out)

    return softmax_calib, y_calib_out, softmax_test, y_test_out
