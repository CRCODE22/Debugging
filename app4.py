# --- Start of Neural Identity Matrix V24.35 (Modified) ---
# Run `python -m py_compile Neural_Identity_Matrix_updated_V24.35.py` to check syntax before execution
# Ensure dataset.csv, previous_names.csv, upper_clothing.csv, lower_clothing.csv, footwear.csv, style_themes.csv, locations.csv, overall_themes.csv are in the project directory
# Setup: conda activate neural-identity-matrix; pip install -r requirements.txt
# Note: Compatible with torch-2.5.1+cu124; update torch.amp for future versions
# Gradio table requires horizontal scrolling for all columns; adjust screen resolution if needed
# ComfyUI must be running locally for image generation
# X API credentials required for sharing feature; set up in environment variables
import gradio as gr
print(f"Gradio version: {gr.__version__}")
# Custom CSS for responsive DataFrame
custom_css = """
.full-width-dataframe {
    width: 100% !important;
    max-width: 100% !important;
    overflow-x: auto !important;
    overflow-y: auto !important;
    max-height: 400px !important;
    padding: 10px !important;
    box-sizing: border-box !important;
    display: block !important;  /* Ensure full width */
}
.full-width-dataframe table {
    width: 100% !important;
    table-layout: auto !important;  /* Allow columns to size naturally */
}
@media (max-width: 1920px) {
    .full-width-dataframe {
        font-size: 0.9em;
    }
}
@media (max-width: 1366px) {
    .full-width-dataframe {
        font-size: 0.8em;
    }
}
"""
import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import random
import glob
os.environ["HF_HUB_OFFLINE"] = "1"
# import pickle # Not strictly needed with torch.save/load for models
import time
import sys
import json
import requests
from PIL import Image
import io
import secrets
import tweepy
def verify_required_files():
    required_files = [
        'dataset.csv', 'previous_names.csv', 'upper_clothing.csv', 
        'lower_clothing.csv', 'footwear.csv', 'style_themes.csv', 
        'locations.csv', 'overall_themes.csv'
    ]
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"Error: The following required files are missing: {', '.join(missing_files)}")
        print("Attempting to create default files for missing CSVs...")
        for file in missing_files:
            if file == 'dataset.csv':
                # Create a minimal dataset.csv
                default_data = pd.DataFrame({
                    'Firstname': ['Ava'], 'Lastname': ['Smith'], 'Nickname': ['Star42'],
                    'Age': [25], 'Height': [165], 'Weight': [60], 'Body Measurements': ['32-26-34'],
                    'Nationality': ['Unknown'], 'Ethnicity': ['Unknown'], 'Birthplace': ['Unknown'],
                    'Profession': ['Astrologer'], 'Body type': ['Average'], 'Hair color': ['Brown'],
                    'Eye color': ['Blue'], 'Bra/cup size': ['B'], 'Boobs': ['Natural']
                })
                default_data.to_csv(file, index=False)
                print(f"Created default {file}")
            elif file == 'previous_names.csv':
                pd.DataFrame({'Firstname': [], 'Lastname': []}).to_csv(file, index=False)
                print(f"Created empty {file}")
            else:
                # For clothing and themes, create minimal CSVs
                pd.DataFrame({'Clothing' if 'clothing' in file or 'footwear' in file else 'Theme': ['Default']}).to_csv(file, index=False)
                print(f"Created default {file}")
        return False
    return True

# --- MODIFICATION START ---
# Configuration
COMFYUI_URL = "http://127.0.0.1:8188" # Make ComfyUI URL configurable

FIRST_NAME_GEN_MODEL_PATH = 'first_name_gen_model.pth'
LAST_NAME_GEN_MODEL_PATH = 'last_name_gen_model.pth'
NICKNAME_GEN_MODEL_PATH = 'nickname_gen_model.pth'
# --- MODIFICATION END ---

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Startup message
print(f"Starting Neural Identity Matrix V24.35 | Device: {device} | Python: {sys.version.split()[0]} | PyTorch: {torch.__version__} | Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Predefined lists (unchanged)
predefined_first_names = [
    'Ava', 'Emma', 'Olivia', 'Sophia', 'Isabella', 'Mia', 'Charlotte', 'Amelia', 'Harper', 'Evelyn',
    'Luna', 'Aria', 'Ella', 'Nora', 'Hazel', 'Zoe', 'Lily', 'Ellie', 'Violet', 'Grace',
    'James', 'Liam', 'Noah', 'William', 'Henry', 'Oliver', 'Elijah', 'Lucas', 'Mason', 'Logan',
    'Ethan', 'Jack', 'Aiden', 'Carter', 'Daniel', 'Owen', 'Wyatt', 'John', 'David', 'Gabriel'
]
predefined_last_names = [
    'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez',
    'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin',
    'Lee', 'Perez', 'Thompson', 'White', 'Harris', 'Sanchez', 'Clark', 'Ramirez', 'Lewis', 'Walker',
    'Hall', 'Allen', 'Young', 'King', 'Wright', 'Scott', 'Green', 'Baker', 'Adams', 'Nelson'
]

# Predefined lists for cosmic playlists
artist_names_list = [
    'Luna Vox', 'Stellar Echo', 'Nebula Drift', 'Aether Pulse', 'Cosmic Weaver',
    'Eclipse Singer', 'Quantum Muse', 'Starlight Bard', 'Galactic Siren', 'Void Whisperer'
]
genres_list = [
    'Ambient', 'Synthwave', 'Ethereal', 'Chillout', 'Downtempo',
    'Trance', 'Psychedelic', 'Cosmic', 'Electronic', 'Dreamscape'
]
themes_list = [
    'Ethereal Drift', 'Stellar Journey', 'Quantum Harmony', 'Nebula Waves', 'Astral Echoes',
    'Lunar Serenity', 'Galactic Dream', 'Void Symphony', 'Starlight Pulse', 'Cosmic Reverie'
]

# Load clothing and theme datasets
if not verify_required_files():
    print("Warning: Some files were missing and replaced with defaults. Functionality may be limited.")
upper_clothing_df = pd.read_csv('upper_clothing.csv')
lower_clothing_df = pd.read_csv('lower_clothing.csv')
footwear_df = pd.read_csv('footwear.csv')
style_themes_df = pd.read_csv('style_themes.csv')
locations_df = pd.read_csv('locations.csv')
overall_themes_df = pd.read_csv('overall_themes.csv')
print(f"Loaded upper_clothing.csv with {len(upper_clothing_df)} items")
print(f"Loaded lower_clothing.csv with {len(lower_clothing_df)} items")
print(f"Loaded footwear.csv with {len(footwear_df)} items")
print(f"Loaded style_themes.csv with {len(style_themes_df)} themes")
print(f"Loaded locations.csv with {len(locations_df)} themes")
print(f"Loaded overall_themes.csv with {len(overall_themes_df)} themes")
def get_theme_column(df, file_name):
    possible_columns = ['Theme', 'theme', 'Style', 'Location', 'Overall Theme']
    for col in possible_columns:
        if col in df.columns:
            return df[col].tolist()
    print(f"Error: No valid theme column found in {file_name}. Available columns: {list(df.columns)}")
    return ['Default']
style_themes_list = get_theme_column(style_themes_df, 'style_themes.csv')
locations_list = get_theme_column(locations_df, 'locations.csv')
overall_themes_list = get_theme_column(overall_themes_df, 'overall_themes.csv')

# --- End of Section 1 ---
# --- Start of Section 2 ---

# Neural Network Model
class IdentityGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(IdentityGenerator, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, (hidden, cell) = self.lstm(x, (hidden, cell))
        out = self.fc(out)
        return out, hidden, cell

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))

# Name Generator
class NameGenerator(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim, num_layers=1):
        super(NameGenerator, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden, cell):
        embedded = self.embedding(x)
        out, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        out = self.fc(out)
        return out, hidden, cell

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))

# Load dataset
if not verify_required_files():
    print("Warning: Some files were missing and replaced with defaults. Functionality may be limited.")
df = pd.read_csv('dataset.csv')
print(f"Loaded dataset.csv with {len(df)} rows")
try:
    additional_names = pd.read_csv('previous_names.csv')
    print(f"Loaded {len(additional_names)} additional first names and last names from previous_names.csv")
except FileNotFoundError:
    print("Warning: previous_names.csv not found. Creating empty DataFrame.")
    additional_names = pd.DataFrame(columns=['Firstname', 'Lastname'])

# Combine dataset names
first_names = list(set(df['Firstname'].tolist() + additional_names['Firstname'].tolist() + predefined_first_names))
last_names = list(set(df['Lastname'].tolist() + additional_names['Lastname'].tolist() + predefined_last_names))
nicknames = df['Nickname'].tolist()

# Build character vocab
first_name_chars = set(''.join(str(name) for name in first_names if pd.notna(name)))
last_name_chars = set(''.join(str(name) for name in last_names if pd.notna(name)))
nickname_chars = set(''.join(str(name) for name in nicknames if pd.notna(name)))

first_name_chars.add('\n')
last_name_chars.add('\n')
nickname_chars.add('\n')

first_name_char_to_idx = {char: idx for idx, char in enumerate(sorted(first_name_chars))}
last_name_char_to_idx = {char: idx for idx, char in enumerate(sorted(last_name_chars))}
nickname_char_to_idx = {char: idx for idx, char in enumerate(sorted(nickname_chars))}

first_name_idx_to_char = {idx: char for char, idx in first_name_char_to_idx.items()}
last_name_idx_to_char = {idx: char for char, idx in last_name_char_to_idx.items()}
nickname_idx_to_char = {idx: char for char, idx in nickname_char_to_idx.items()}

# Hyperparameters
hidden_size = 256
embedding_dim = 64
num_layers = 1
first_name_max_len = max(len(str(name)) for name in first_names if pd.notna(name)) + 1
last_name_max_len = max(len(str(name)) for name in last_names if pd.notna(name)) + 1
nickname_max_len = 20


# Training name generators (original function)
def train_name_generator_core(model, names, char_to_idx, max_len, epochs=100): # Renamed to core
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        name_count = 0
        for name in names:
            if pd.isna(name) or not name: # Ensure name is not NaN and not empty
                continue
            name_str = str(name) + '\n'
            if not name_str.strip(): # Skip if name is just whitespace after becoming a string
                continue

            try:
                inputs_list = [char_to_idx[char] for char in name_str[:-1]]
                targets_list = [char_to_idx[char] for char in name_str[1:]]
            except KeyError as e:
                print(f"Warning: Character {e} not in vocab for name '{name_str.strip()}'. Skipping this name for training.")
                continue
            
            if not inputs_list: # Skip if inputs list is empty (e.g. single char name that's not in vocab)
                continue

            inputs = torch.tensor(inputs_list, dtype=torch.long).unsqueeze(0).to(device)
            targets = torch.tensor(targets_list, dtype=torch.long).unsqueeze(0).to(device)

            hidden, cell = model.init_hidden(1)
            optimizer.zero_grad()

            outputs, hidden, cell = model(inputs, hidden, cell)
            loss = criterion(outputs.squeeze(0), targets.squeeze(0)) # Adjusted squeeze for batch size 1
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            name_count +=1
        
        if name_count == 0:
            print(f"Epoch {(epoch + 1)}/{epochs}, No valid names to train on.")
            continue

        if (epoch + 1) % 20 == 0:
            print(f'Epoch {(epoch + 1)}/{epochs}, Loss: {total_loss / name_count:.4f}')

# --- MODIFICATION START ---
# Initialize name generators
first_name_gen = NameGenerator(len(first_name_char_to_idx), hidden_size, embedding_dim, num_layers).to(device)
last_name_gen = NameGenerator(len(last_name_char_to_idx), hidden_size, embedding_dim, num_layers).to(device)
nickname_gen = NameGenerator(len(nickname_char_to_idx), hidden_size, embedding_dim, num_layers).to(device)

# Load or train name generator models
def load_or_train_name_model(model, model_path, model_name_for_log, names_data, char_to_idx_map, max_name_len, epochs=100):
    if os.path.exists(model_path):
        print(f"Loading saved {model_name_for_log} generator model from {model_path}")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.to(device) # Ensure model is on the correct device
            model.eval() # Set to evaluation mode
            print(f"Successfully loaded {model_name_for_log} model.")
        except Exception as e:
            print(f"Error loading {model_name_for_log} model from {model_path}: {e}. Retraining...")
            train_name_generator_core(model, names_data, char_to_idx_map, max_name_len, epochs=epochs)
            torch.save(model.state_dict(), model_path)
            print(f"{model_name_for_log} model trained and saved to {model_path}.")
    else:
        print(f"{model_name_for_log} model not found at {model_path}. Training new model...")
        train_name_generator_core(model, names_data, char_to_idx_map, max_name_len, epochs=epochs)
        torch.save(model.state_dict(), model_path)
        print(f"{model_name_for_log} model trained and saved to {model_path}.")
    return model

print("Initializing or loading name generator models...")
# Filter out NaN before passing to training to prevent issues with `str(name)` for NaN values.
# Also ensure names are not empty after potential stripping if that logic is applied before training.
valid_first_names = [name for name in first_names if pd.notna(name) and str(name).strip()]
valid_last_names = [name for name in last_names if pd.notna(name) and str(name).strip()]
valid_nicknames = [name for name in nicknames if pd.notna(name) and str(name).strip()]


if valid_first_names and first_name_char_to_idx : # Ensure there's data to train on
    first_name_gen = load_or_train_name_model(first_name_gen, FIRST_NAME_GEN_MODEL_PATH, "First Name", valid_first_names, first_name_char_to_idx, first_name_max_len)
else:
    print("Warning: No valid first names or character vocabulary for first name generator. Skipping training/loading.")

if valid_last_names and last_name_char_to_idx:
    last_name_gen = load_or_train_name_model(last_name_gen, LAST_NAME_GEN_MODEL_PATH, "Last Name", valid_last_names, last_name_char_to_idx, last_name_max_len)
else:
    print("Warning: No valid last names or character vocabulary for last name generator. Skipping training/loading.")

if valid_nicknames and nickname_char_to_idx:
    nickname_gen = load_or_train_name_model(nickname_gen, NICKNAME_GEN_MODEL_PATH, "Nickname", valid_nicknames, nickname_char_to_idx, nickname_max_len)
else:
    print("Warning: No valid nicknames or character vocabulary for nickname generator. Skipping training/loading.")

# --- MODIFICATION END ---


# Nickname suffixes
nickname_suffixes = [
    'Star', 'Cosmo', 'Dreamer', 'Vibe', 'Guru', 'Nebula', 'Quantum', 'Spark', '42',
    'Player', 'GamerX', 'Pro', 'ModelX', 'Starlet', 'Glam', 'Clone', 'NIM', 'Core'
]
print(f"Nickname settings: min_length=3, max_length={nickname_max_len}, suffixes={nickname_suffixes}")

# Generate names
def generate_name(generator, char_to_idx, idx_to_char, max_len, device, name_type='firstname', existing_names=None, temperature=0.7):
    if not char_to_idx or not idx_to_char : # Check if vocab is empty
        print(f"Warning: Character vocabulary for {name_type} is empty. Cannot generate name.")
        if name_type == 'firstname': return random.choice(predefined_first_names) if predefined_first_names else "DefaultFirst"
        if name_type == 'lastname': return random.choice(predefined_last_names) if predefined_last_names else "DefaultLast"
        return f"Nick{name_type.capitalize()}"

    generator.eval()
    with torch.no_grad():
        for attempt in range(20): # Increased attempts
            name_chars = [] # Renamed from 'name' to 'name_chars' to avoid confusion
            
            current_existing_names = existing_names if existing_names is not None else set()

            # Determine valid starting characters
            if name_type in ['firstname', 'lastname'] and current_existing_names:
                 # Use first characters from the provided names dataset if available and valid
                valid_starts_from_data = [n[0] for n in (first_names if name_type == 'firstname' else last_names) if n and isinstance(n, str) and len(n) > 0 and n[0] in char_to_idx]
                if not valid_starts_from_data: # Fallback if no valid starts from data
                    valid_starts_from_data = [c for c in char_to_idx.keys() if c.isalpha() and c.isupper()] # Prefer uppercase letters
                start_char = random.choice(valid_starts_from_data if valid_starts_from_data else list(char_to_idx.keys() - {'\n'}))
            else: # For nicknames or if no existing names for first/last
                valid_starts = list(char_to_idx.keys() - {'\n'}) # Exclude newline as a start
                start_char = random.choice(valid_starts if valid_starts else ['A']) # Fallback to 'A'

            if start_char not in char_to_idx: # Final check if start_char somehow invalid
                # print(f"Warning: Start character '{start_char}' not in char_to_idx for {name_type}. Using a fallback.")
                fallback_chars = list(char_to_idx.keys() - {'\n'})
                start_char = random.choice(fallback_chars) if fallback_chars else list(char_to_idx.keys())[0]


            name_chars.append(start_char)
            input_char_idx = char_to_idx[start_char]
            input_tensor = torch.tensor([[input_char_idx]], dtype=torch.long).to(device)
            
            hidden, cell = generator.init_hidden(1)
            
            min_length = 3 if name_type == 'nickname' else 2 # Min length for first/last names is 2

            for _ in range(max_len -1): # -1 because start_char is already added
                output, hidden, cell = generator(input_tensor, hidden, cell)
                output_dist = output.squeeze().div(temperature).exp()
                
                # Avoid generating newline if name is too short
                if len(name_chars) < min_length and '\n' in idx_to_char.values():
                    forbidden_indices = {char_to_idx['\n']}
                    # Create a mask to zero out forbidden characters
                    mask = torch.ones_like(output_dist)
                    for idx_to_forbid in forbidden_indices:
                        if 0 <= idx_to_forbid < len(mask): # Check bounds
                             mask[idx_to_forbid] = 0
                    output_dist = output_dist * mask
                    if torch.sum(output_dist) == 0: # if all valid chars masked, break
                        break 


                char_idx = torch.multinomial(output_dist, 1).item()
                char = idx_to_char[char_idx]
                
                if char == '\n':
                    if len(name_chars) >= min_length:
                        break
                    else: # Name too short, try to generate another character instead of ending
                        continue 
                
                name_chars.append(char)
                input_tensor = torch.tensor([[char_idx]], dtype=torch.long).to(device)
            
            generated_name_str = ''.join(name_chars).replace('\n', '').capitalize()
            
            if name_type == 'nickname' and random.random() < 0.5:
                suffix = random.choice(nickname_suffixes)
                generated_name_str += suffix
            
            # Validation checks
            invalid_chars_set = set(',- ') # Allow spaces in nicknames but not start/end
            if name_type != 'nickname':
                 generated_name_str = generated_name_str.replace(' ', '')


            if (len(generated_name_str) < min_length or
                (name_type != 'nickname' and any(char in invalid_chars_set for char in generated_name_str)) or
                (name_type == 'nickname' and (generated_name_str.startswith(' ') or generated_name_str.endswith(' '))) or
                any(char.lower() not in char_to_idx and char != '\n' for char in generated_name_str)): # check against original vocab
                # print(f"Debug: Invalid generated name '{generated_name_str}' for {name_type}. Retrying. Length: {len(generated_name_str)}, MinLength: {min_length}")
                continue
            
            if current_existing_names and generated_name_str in current_existing_names:
                # print(f"Debug: Name '{generated_name_str}' already exists in {name_type}. Retrying.")
                continue

            return generated_name_str
        
        # Fallback if too many attempts fail
        # print(f"Warning: Could not generate a unique, valid {name_type} after {attempt+1} attempts. Using a predefined or placeholder name.")
        if name_type == 'firstname':
            return random.choice(predefined_first_names) if predefined_first_names else "DefaultFirst"
        elif name_type == 'lastname':
            return random.choice(predefined_last_names) if predefined_last_names else "DefaultLast"
        else: # nickname
            base_nick = f"Nick{random.randint(100,999)}"
            if random.random() < 0.5: base_nick += random.choice(nickname_suffixes)
            return base_nick


# --- End of Section 2 ---
# --- Start of Section 3 ---

# Preprocess data
le_dict = {}
for column in ['Nationality', 'Ethnicity', 'Birthplace', 'Profession', 'Body type', 'Hair color', 'Eye color', 'Bra/cup size', 'Boobs']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    le_dict[column] = le

new_professions = ['Astrologer', 'Chef', 'DJ', 'Engineer', 'Gamer', 'Hacker', 'Pilot', 'Scientist', 'Streamer', 'Writer', 'High Priestess Witch']
df['Profession'] = le_dict['Profession'].inverse_transform(df['Profession'])
print("Types in 'Profession' column before adding new professions:", df['Profession'].apply(type).unique())
for prof in new_professions:
    if prof not in df['Profession'].values:
        new_row = df.iloc[0].copy()
        new_row['Profession'] = prof
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
print("Types in 'Profession' column after adding new professions:", df['Profession'].apply(type).unique())
le_dict['Profession'] = LabelEncoder()
df['Profession'] = le_dict['Profession'].fit_transform(df['Profession'])
print(f"Updated professions: {le_dict['Profession'].classes_}")

scaler_age = StandardScaler()
scaler_height = StandardScaler()
scaler_weight = StandardScaler()
scaler_measurements = StandardScaler()
scaler_features = StandardScaler()

df['Age'] = scaler_age.fit_transform(df[['Age']])
df['Height'] = scaler_height.fit_transform(df[['Height']])
df['Weight'] = scaler_weight.fit_transform(df[['Weight']])

body_measurements = df['Body Measurements'].str.split('-', expand=True).astype(float)
df[['Bust', 'Waist', 'Hips']] = scaler_measurements.fit_transform(body_measurements)

features_cols = ['Age', 'Height', 'Weight', 'Bust', 'Waist', 'Hips', 'Nationality', 'Ethnicity', 'Birthplace', 'Profession', 'Body type', 'Hair color', 'Eye color', 'Bra/cup size', 'Boobs']
features = df[features_cols].values
features = scaler_features.fit_transform(features)


# Initialize model
input_size = features.shape[1]
output_size = features.shape[1]
model = IdentityGenerator(input_size, hidden_size, output_size, num_layers).to(device)

# Training loop
def train_model(model, features_data, cycles=5, epochs_per_cycle=20, verbose=False): # Renamed features to features_data
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # --- MODIFICATION START ---
    # Ensure GradScaler is only enabled if CUDA is available and being used
    use_amp = device.type == 'cuda'
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
    # --- MODIFICATION END ---
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epochs_per_cycle // 2, gamma=0.5)
    losses = []
    all_losses = []
    total_epochs = 0
    # Ensure log file can be written, handle potential permission errors
    try:
        log_file = open('training_log.txt', 'w')
    except PermissionError:
        print("Warning: Permission denied to write training_log.txt. Logging to console only.")
        log_file = None


    print(f"Training with {len(features_data)} features, shape: {features_data.shape}")

    try:
        for cycle in range(cycles):
            print(f"Starting Cycle {cycle + 1}/{cycles}")
            cycle_losses = []
            for epoch in range(epochs_per_cycle):
                model.train()
                total_loss_epoch = 0 # Renamed to avoid conflict
                start_time = datetime.now()

                for i in range(0, len(features_data), 1): # Iterate one by one
                    inputs = torch.tensor(features_data[i:i+1], dtype=torch.float32).to(device)
                    targets = inputs.clone()

                    hidden, cell = model.init_hidden(1) # Batch size is 1
                    optimizer.zero_grad()
                    
                    # --- MODIFICATION START ---
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            outputs, hidden, cell = model(inputs, hidden, cell)
                            outputs = outputs.squeeze(1) # Ensure correct shape
                            loss = criterion(outputs, targets)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else: # CPU or CUDA without AMP
                        outputs, hidden, cell = model(inputs, hidden, cell)
                        outputs = outputs.squeeze(1) # Ensure correct shape
                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()
                    # --- MODIFICATION END ---
                    total_loss_epoch += loss.item()

                scheduler.step()
                total_epochs += 1
                avg_loss = total_loss_epoch / len(features_data)
                cycle_losses.append(avg_loss)
                all_losses.append(avg_loss)
                epoch_time = (datetime.now() - start_time).total_seconds()

                log_message = f"Cycle {cycle + 1}/{cycles}, Epoch {epoch + 1}/{epochs_per_cycle} | Avg Loss: {avg_loss:.6f} | Time: {epoch_time:.2f}s | Speed: {1/epoch_time if epoch_time > 0 else float('inf'):.2f} epochs/s | LR: {scheduler.get_last_lr()[0]:.6f}\n"
                if log_file:
                    log_file.write(log_message)
                if verbose or (epoch + 1) % 10 == 0:
                    print(log_message.strip())

                # Early stopping logic (unchanged, but ensure all_losses is populated)
                if len(all_losses) > 10:
                    # Ensure there are enough elements for min/max calculation
                    recent_losses_for_min = all_losses[-11:-1] if len(all_losses) > 1 else all_losses
                    if recent_losses_for_min:
                        min_loss = min(recent_losses_for_min)
                        if avg_loss > min_loss * 1.1: # Increased tolerance slightly
                            print(f"Early stopping triggered (loss increase) at Cycle {cycle + 1}, Epoch {epoch + 1}")
                            if log_file: log_file.write(f"Early stopping (loss increase)\n")
                            if log_file: log_file.close()
                            return all_losses, total_epochs
                    
                    recent_losses_for_plateau = all_losses[-11:-1] if len(all_losses) > 1 else all_losses
                    if recent_losses_for_plateau:
                        max_loss = max(recent_losses_for_plateau)
                        if abs(max_loss - avg_loss) < 1e-6: # Check absolute difference for plateau
                            print(f"Early stopping triggered (loss plateau) at Cycle {cycle + 1}, Epoch {epoch + 1}")
                            if log_file: log_file.write(f"Early stopping (loss plateau)\n")
                            if log_file: log_file.close()
                            return all_losses, total_epochs
                
                if (epoch + 1) % 5 == 0: # Yield for Gradio update
                    yield cycle_losses, total_epochs # cycle_losses is fine, total_epochs tracks overall progress

            losses.extend(cycle_losses)
            yield cycle_losses, total_epochs # Yield after each cycle for Gradio update

        if log_file:
            log_file.close()
        final_loss_val = all_losses[-1] if all_losses else float('nan')
        print(f"Training completed: Total Epochs: {total_epochs}, Final Loss: {final_loss_val:.6f}")
        return all_losses, total_epochs

    except KeyboardInterrupt:
        print("Training interrupted! Saving model state...")
        if log_file:
            log_file.write("Training interrupted\n")
            log_file.close()
        torch.save(model.state_dict(), 'model_interrupted.pth')
        raise
    except Exception as e:
        print(f"An error occurred during training: {e}")
        if log_file:
            log_file.write(f"Error during training: {e}\n")
            log_file.close()
        raise


# --- End of Section 3 ---
# --- Start of Section 4 ---

def check_comfyui_availability(url):
    try:
        response = requests.get(f"{url}/system_stats", timeout=5)
        if response.status_code == 200:
            print("ComfyUI server is running.")
            return True
        else:
            print(f"ComfyUI server responded with status {response.status_code}.")
            return False
    except requests.exceptions.RequestException as e:
        print(f"ComfyUI server is not running at {url}: {e}")
        return False

# Generate unique filename
def generate_unique_filename(base_name):
    output_dir = "generated_images" # Suggest saving to a subfolder
    os.makedirs(output_dir, exist_ok=True)
    while True:
        random_suffix = secrets.token_hex(5).upper()[:11]
        filename = os.path.join(output_dir, f"{base_name}_{random_suffix}.png")
        if not os.path.exists(filename):
            return filename
        print(f"Filename {filename} already exists, generating a new suffix...")


# Generate image with PG/NSFW option, style theme, location, and overall theme
# --- MODIFICATION START ---
def generate_flux_image(selected_identity, df_identities, allow_nsfw=False, style_theme="Cyberpunk", location="Cosmic Nebula", overall_theme="Ethereal Dreamscape", image_seed=0):
    if selected_identity == "None" or df_identities is None or df_identities.empty:
        return None, "Please generate identities and select one for image generation."
    if not check_comfyui_availability(COMFYUI_URL):
        return None, f"Error: ComfyUI server is not running at {COMFYUI_URL}. Please start the server and try again."

    try:
        clone_number, nickname = selected_identity.split(": ")
        row = df_identities[df_identities['Clone Number'] == clone_number].iloc[0]

        # Skip if image already exists
        existing_images = set(os.listdir("generated_images") if os.path.exists("generated_images") else [])
        image_filename = f"{clone_number}_{row['Nickname']}"
        if any(image_filename in img for img in existing_images):
            print(f"Skipping image generation for {image_filename}: already exists")
            return None, f"Image for {selected_identity} already exists."

        # Select clothing
        upper_clothing = random.choice(upper_clothing_df['Clothing'].tolist()) if not upper_clothing_df.empty else "T-shirt"
        lower_clothing = random.choice(lower_clothing_df['Clothing'].tolist()) if not lower_clothing_df.empty else "Jeans"
        footwear = random.choice(footwear_df['Clothing'].tolist()) if not footwear_df.empty else "Sneakers"
        row['Upper Clothing'] = upper_clothing
        row['Lower Clothing'] = lower_clothing
        row['Footwear'] = footwear

        if style_theme == "Real-Life":
            prompt = (
                f"A realistic portrait of a female named {row['Nickname']}, {row['Age']} years old, "
                f"with {row['Hair color'].lower()} hair and {row['Eye color'].lower()} eyes, "
                f"in a modern urban setting, wearing casual clothes like a {upper_clothing.lower()} and {lower_clothing.lower()}, "
                f"with a natural smile, radiating a {row['Energy Signature'].lower()} vibe"
            )
        elif style_theme == "Instagram Selfie":
            prompt = (
                f"A vibrant Instagram-style selfie of a female named {row['Nickname']}, {row['Age']} years old, "
                f"with {row['Hair color'].lower()} hair and {row['Eye color'].lower()} eyes, "
                f"in a trendy cafe, wearing stylish clothes like a {upper_clothing.lower()} and {lower_clothing.lower()}, "
                f"with a playful pose, radiating a {row['Energy Signature'].lower()} vibe"
            )
        elif style_theme == "Cinematic":
            prompt = (
                f"A cinematic shot of a female named {row['Nickname']}, {row['Age']} years old, "
                f"with {row['Hair color'].lower()} hair and {row['Eye color'].lower()} eyes, "
                f"in a dramatic cityscape at dusk, wearing elegant clothes like a {upper_clothing.lower()} and {lower_clothing.lower()}, "
                f"with a confident stance, radiating a {row['Energy Signature'].lower()} vibe"
            )
        else:  # Cyberpunk
            prompt = (
                f"A cinematic shot of a futuristic female clone named {row['Nickname']}, {row['Age']} years old, "
                f"with {row['Hair color'].lower()} hair and {row['Eye color'].lower()} eyes, "
                f"with a {row['Body type'].lower()} build, body measurements {row['Body Measurements']}, "
                f"height {row['Height']} cm, weight {row['Weight']} kg, "
                f"in a {location.lower()} setting, styled in a {style_theme.lower()} aesthetic, "
                f"within an overall {overall_theme.lower()} atmosphere, "
                f"glowing with a {row['Cosmic Aura'].lower() if row['Cosmic Aura'] != 'None' else 'electric starlight'} aura, "
                f"radiating a {row['Energy Signature'].lower()} energy, "
                f"adorned with a {row['Cosmic Tattoo'].lower() if row['Cosmic Tattoo'] != 'None' else 'subtle cosmic pattern'} tattoo, "
                f"accompanied by a {row['Cosmic Pet'].lower() if row['Cosmic Pet'] != 'None' else 'faint cosmic sparkle'}, "
                f"embodying the destiny of a {row['Cosmic Destiny'].lower() if row['Cosmic Destiny'] != 'None' else 'stellar traveler'}, "
                f"wearing a {upper_clothing.lower()}, {lower_clothing.lower()}, and {footwear.lower()}"
            )
        if not allow_nsfw:
            print(f"Generating FLUX.1 [dev] image for {selected_identity} with PG-rated prompt: {prompt}")
            print(f"DEBUG: Selected clothing - Upper: {upper_clothing}, Lower: {lower_clothing}, Footwear: {footwear}")
        else:
            prompt += ", potentially NSFW, may include nudity or suggestive elements"
            print(f"Generating FLUX.1 [dev] image for {selected_identity} with NSFW prompt: {prompt}")
        
        print(f"DEBUG: Style Theme: {style_theme}, Location: {location}, Overall Theme: {overall_theme}")
        
        # --- MODIFICATION START ---
        actual_seed = int(time.time()) if image_seed == 0 else int(image_seed)
        print(f"Using seed: {actual_seed} for image generation.")
        # --- MODIFICATION END ---

        workflow = {
            "9": {
                "inputs": {"filename_prefix": f"{clone_number}_{nickname}", "images": ["8", 0]},
                "class_type": "SaveImage"
            },
            "8": {
                "inputs": {"samples": ["13", 0], "vae": ["10", 0]},
                "class_type": "VAEDecode"
            },
            "10": {
                "inputs": {"vae_name": "ae.safetensors"},
                "class_type": "VAELoader"
            },
            "13": {
                "inputs": {
                    "noise": ["25", 0],
                    "guider": ["22", 0],
                    "sampler": ["16", 0],
                    "sigmas": ["17", 0],
                    "latent_image": ["41", 0]
                },
                "class_type": "SamplerCustomAdvanced"
            },
            "16": {
                "inputs": {"sampler_name": "euler"},
                "class_type": "KSamplerSelect"
            },
            "17": {
                "inputs": {"model": ["30", 0], "scheduler": "beta", "steps": 30, "denoise": 1.0},
                "class_type": "BasicScheduler"
            },
            "22": {
                "inputs": {"model": ["63", 0], "conditioning": ["26", 0]},
                "class_type": "BasicGuider"
            },
            "25": { # RandomNoise node
                # --- MODIFICATION START ---
                "inputs": {"noise_seed": actual_seed, "width": 768, "height": 768},
                # --- MODIFICATION END ---
                "class_type": "RandomNoise"
            },
            "26": {
                "inputs": {"conditioning": ["45", 0], "guidance": 3.5},
                "class_type": "FluxGuidance"
            },
            "30": {
                "inputs": {
                    "model": ["12", 0],
                    "width": 768,
                    "height": 768,
                    "max_shift": 1.15,
                    "base_shift": 0.5
                },
                "class_type": "ModelSamplingFlux"
            },
            "41": {
                "inputs": {"width": 768, "height": 768, "batch_size": 1},
                "class_type": "EmptyLatentImage"
            },
            "45": {
                "inputs": {"text": prompt, "clip": ["63", 1]},
                "class_type": "CLIPTextEncode"
            },
            "59": {
                "inputs": {
                    "clip_name1": "t5xxl_fp16.safetensors",
                    "clip_name2": "godessProjectFLUX_clipLFP8.safetensors",
                    "clip_name3": "clip_g.safetensors"
                },
                "class_type": "TripleCLIPLoader"
            },
            "63": {
                "inputs": {
                    "model": ["12", 0],
                    "clip": ["59", 0],
                    "lora_01": "None", "strength_01": 0.0, "lora_02": "None", "strength_02": 0.0,
                    "lora_03": "None", "strength_03": 0.0, "lora_04": "None", "strength_04": 0.0
                },
                "class_type": "Lora Loader Stack (rgthree)"
            },
            "12": {
                "inputs": {"unet_name": "acornIsSpinningFLUX_devfp8V11.safetensors", "weight_dtype": "fp8_e4m3fn"},
                "class_type": "UNETLoader"
            }
        }

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- MODIFICATION START ---
        response = requests.post(
            f"{COMFYUI_URL}/prompt", # Use configured URL
            json={"prompt": workflow},
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        # --- MODIFICATION END ---
        response.raise_for_status()
        result = response.json()
        prompt_id = result.get("prompt_id")
        print(f"Submitted prompt with ID: {prompt_id}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Cleared CUDA memory after submitting prompt")

        base_name = nickname.replace(' ', '').lower()
        # Save to a subfolder called "generated_images"
        output_path = generate_unique_filename(base_name) # Already creates in subfolder

        start_time_poll = time.time() # Renamed to avoid conflict
        while time.time() - start_time_poll < 900: # 15 minutes timeout
            # --- MODIFICATION START ---
            history_response = requests.get(f"{COMFYUI_URL}/history/{prompt_id}")
            # --- MODIFICATION END ---
            history_response.raise_for_status() # Check for errors in history request
            history = history_response.json()

            if prompt_id in history and history[prompt_id].get("status", {}).get("completed", False):
                outputs_data = history[prompt_id].get("outputs", {})
                image_node_output = outputs_data.get("9", {}).get("images") # Node "9" is SaveImage
                
                if image_node_output and len(image_node_output) > 0:
                    image_info = image_node_output[0]
                    image_filename = image_info.get("filename")
                    image_subfolder = image_info.get("subfolder", "")
                    image_type = image_info.get("type", "output") # Usually "output" or "temp"
                    
                    # --- MODIFICATION START ---
                    # Construct URL to view/fetch the image
                    # Ensure subfolder is handled correctly, it might be empty
                    view_url_path = f"view?filename={requests.utils.quote(image_filename)}"
                    if image_subfolder:
                         view_url_path += f"&subfolder={requests.utils.quote(image_subfolder)}"
                    view_url_path += f"&type={image_type}"
                    image_url = f"{COMFYUI_URL}/{view_url_path}"
                    # --- MODIFICATION END ---
                    
                    print(f"Fetching image from ComfyUI: {image_url}")
                    image_response = requests.get(image_url, timeout=60)
                    image_response.raise_for_status() # Check for errors fetching image

                    if image_response.content:
                        image = Image.open(io.BytesIO(image_response.content))
                        image.save(output_path) # Save to our uniquely generated path
                        print(f"Image saved as {output_path}")
                        return output_path, f"Image generated successfully for {selected_identity}."
                    else:
                        print("Error: Empty image content received from ComfyUI.")
                        return None, "Error: Empty image content from ComfyUI."
                else:
                    print(f"Error: No image data found in output node '9' for prompt {prompt_id}. Outputs: {outputs_data}")
                    # Try to find any image in any output node if primary fails (more robust)
                    for node_id, node_data in outputs_data.items():
                        if "images" in node_data and len(node_data["images"]) > 0:
                            print(f"Found image in alternative node '{node_id}'. Using this.")
                            # (Logic to process alternative node similar to above)
                            # This part can be expanded if needed. For now, error out.
                            break # Found an image, process it (not fully implemented here for brevity)
                    return None, "Error: No image data found in expected workflow output node."
            time.sleep(2)

        print("Image generation timed out after 15 minutes.")
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        with open('error_log.txt', 'a') as f:
            f.write(f"{datetime.now()}: Image generation timed out for {selected_identity}\n")
        return None, "Image generation timed out after 15 minutes."

    except requests.exceptions.RequestException as e:
        err_msg = f"ComfyUI API error: {str(e)}"
        if 'response' in locals() and hasattr(response, 'text'): err_msg += f" | API Response: {response.text[:500]}" # Show partial response
        print(err_msg)
        with open('error_log.txt', 'a') as f: f.write(f"{datetime.now()}: {err_msg} for {selected_identity}\n")
        return None, f"ComfyUI API error. Check console/logs. Is ComfyUI running at {COMFYUI_URL}?"
    except Exception as e:
        err_msg = f"Unexpected error in generate_flux_image: {str(e)} (Type: {type(e).__name__})"
        print(err_msg)
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        with open('error_log.txt', 'a') as f: f.write(f"{datetime.now()}: Error generating image for {selected_identity}: {err_msg}\nTrace: {traceback.format_exc()}\n")
        return None, f"Unexpected error: {str(e)}. Check console/logs."
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Batch image generation with new options
def generate_images_batch(df_identities, batch_size=10, allow_nsfw=False, style_theme="Cyberpunk", location="Cosmic Nebula", overall_theme="Ethereal Dreamscape"): # Seed not added to batch yet
    if df_identities is None or df_identities.empty:
        yield None, "No identities available for image generation.", ["No images generated yet."], 0
        return

    total_identities = len(df_identities)
    generated_image_paths = [] # Store paths of successfully generated images for the gallery
    print(f"Starting batch image generation for {total_identities} identities, batch size: {batch_size}, NSFW: {allow_nsfw}, Style: {style_theme}, Location: {location}, Overall: {overall_theme}")

    for start_idx in range(0, total_identities, batch_size):
        end_idx = min(start_idx + batch_size, total_identities)
        current_batch_df = df_identities.iloc[start_idx:end_idx] # Use a different variable name
        print(f"Processing batch {start_idx // batch_size + 1}, identities {start_idx + 1}-{end_idx} of {total_identities}")

        for _, row in current_batch_df.iterrows():
            selected_identity = f"{row['Clone Number']}: {row['Nickname']}"
            print(f"Generating image for {selected_identity} in batch")
            # Batch currently uses random seed for each image by default (image_seed=0)
            image_path, status = generate_flux_image(selected_identity, df_identities, allow_nsfw, style_theme, location, overall_theme, image_seed=0)
            if image_path:
                print(f"Batch image generated: {image_path}")
                generated_image_paths.append(image_path)
            else:
                print(f"Batch image failed for {selected_identity}: {status}")
        
        # Update gallery with all images found so far (could be inefficient for very large batches)
        # For now, pass the accumulated list. A better way might be to update gallery incrementally.
        gallery_images_to_display = display_image_gallery(df_identities) # This re-scans, could use generated_image_paths
        progress = (end_idx / total_identities) * 100
        yield None, f"Batch {start_idx // batch_size + 1} processed. Generated images for {end_idx}/{total_identities} selected identities.", gallery_images_to_display, progress
        time.sleep(1) # Small delay between batches

    final_gallery_images = display_image_gallery(df_identities) # Final scan
    yield None, "Batch image generation complete.", final_gallery_images, 100


# Display image gallery
def display_image_gallery(df_identities): # df_identities might not be needed if scanning a generic folder
    print("DEBUG: Entering display_image_gallery")
    image_folder = "generated_images" # Consistent with generate_unique_filename
    if not os.path.isdir(image_folder):
        print(f"DEBUG: Image folder '{image_folder}' not found.")
        return ["No images generated yet or image folder missing."]

    image_paths = []
    # Simpler scan: just get all PNGs from the folder, sort by modification time (newest first)
    # This avoids needing df_identities if we just want to show what's in the folder.
    try:
        all_files_in_folder = glob.glob(os.path.join(image_folder, "*.png"))
        # Sort by modification time, newest first
        all_files_in_folder.sort(key=os.path.getmtime, reverse=True)
        image_paths = all_files_in_folder
        print(f"DEBUG: Found {len(image_paths)} images in {image_folder}")
    except Exception as e:
        print(f"Error scanning image gallery: {e}")
        return ["Error loading image gallery."]


    if not image_paths:
        print("DEBUG: No images found in folder, returning default message")
        return ["No images generated yet in the 'generated_images' folder."]
    
    print(f"DEBUG: Returning {len(image_paths)} images for gallery")
    return image_paths


# Add the new generate_audio_prompt function here:
def generate_audio_prompt(selected_identity, df_identities, style_theme, location, overall_theme):
    if selected_identity == "None" or df_identities is None or df_identities.empty:
        return "Please generate identities and select one to create an audio prompt."

    try:
        clone_number, nickname = selected_identity.split(": ")
        row = df_identities[df_identities['Clone Number'] == clone_number].iloc[0]
        playlist = row['Cosmic Playlist']
        if playlist == 'None':
            return "This clone has no playlist. Please select a clone with a cosmic playlist."
        
        # Attempt to extract genre more robustly
        playlist_parts = playlist.split()
        genre = "ambient" # Default genre
        for part in playlist_parts:
            if part.lower() in [g.lower() for g in genres_list]: # Check against known genres
                genre = part.lower()
                break
        
        cosmic_elements = []
        if row['Cosmic Aura'] != 'None': cosmic_elements.append(f"echoing the {row['Cosmic Aura'].lower()}")
        if row['Cosmic Destiny'] != 'None': cosmic_elements.append(f"reflecting the destiny of a {row['Cosmic Destiny'].lower()}")
        if row['Cosmic Ability'] != 'None': cosmic_elements.append(f"infused with the power of {row['Cosmic Ability'].lower()}")
        cosmic_desc = ", ".join(cosmic_elements) if cosmic_elements else "with a celestial vibe"
        
        prompt = (
            f"A {genre} track with {overall_theme.lower()} tones, "
            f"inspired by a {location.lower()} setting and {playlist}. "
            f"The music should evoke the clone's {row['Energy Signature'].lower()} energy, "
            f"{cosmic_desc}."
        )
        print(f"Generated audio prompt for {row['Nickname']}: {prompt}")
        return prompt
    except IndexError: # If split fails or row not found
         error_msg = f"Error generating audio prompt: Could not find selected identity or parse playlist."
         print(error_msg)
         with open('error_log.txt', 'a') as f: f.write(f"{datetime.now()}: {error_msg}\n")
         return error_msg
    except Exception as e:
        error_msg = f"Error generating audio prompt: {str(e)}"
        print(error_msg)
        with open('error_log.txt', 'a') as f: f.write(f"{datetime.now()}: {error_msg}\n")
        return error_msg

def suggest_caption(row):
    # Ensure row is a Series, not DataFrame
    if isinstance(row, pd.DataFrame):
        if not row.empty:
            row = row.iloc[0]
        else:
            return "Generating caption... #AI #Art"


    traits = [
        f"{row.get('Nickname', 'A clone')}, a {row.get('Profession', 'mysterious figure').lower()}",
        f"with {row.get('Hair color', 'unique').lower()} hair",
        f"glowing with {row.get('Cosmic Aura', 'electric starlight').lower() if row.get('Cosmic Aura') != 'None' else 'electric starlight'}",
        f"destined as a {row.get('Cosmic Destiny', 'stellar traveler').lower() if row.get('Cosmic Destiny') != 'None' else 'stellar traveler'}"
    ]
    if row.get('Cosmic Pet') != 'None': traits.append(f"with her {row['Cosmic Pet'].lower()}")
    if row.get('Cosmic Hobby') != 'None': traits.append(f"enjoying {row['Cosmic Hobby'].lower()}")
    
    # Add a fallback for profession if it's an integer (not inverse transformed)
    profession_val = row.get('Profession')
    if isinstance(profession_val, (int, np.integer)):
        try:
            profession_val = le_dict['Profession'].inverse_transform([profession_val])[0]
        except:
            profession_val = "unknown profession" # Fallback

    return f"{random.choice(traits) if traits else 'An amazing AI creation'} shines in V24.35! ✨ #CosmicClones #AIArt #NeuralIdentityMatrix"


# Share image to X with suggested caption
def share_to_x(image_path_input, caption, df_identities, selected_identity):
    # Validate inputs
    image_path = None
    if isinstance(image_path_input, str) and os.path.exists(image_path_input):
        image_path = image_path_input
    elif isinstance(image_path_input, dict) and 'path' in image_path_input and os.path.exists(image_path_input['path']): # Gradio image component sometimes returns a dict
        image_path = image_path_input['path']
    elif isinstance(image_path_input, list) and image_path_input and isinstance(image_path_input[0], str) and os.path.exists(image_path_input[0]): # Gallery output
        image_path = image_path_input[0]


    if not image_path:
        return "Error: Image path is invalid or image does not exist. Please generate and select an image first."
    if not selected_identity or selected_identity == "None":
        return "Error: Please select an identity first."
    if df_identities is None or df_identities.empty:
        return "Error: Identities data is missing."

    try:
        clone_number, nickname = selected_identity.split(": ")
        # Ensure df_identities is a DataFrame
        if not isinstance(df_identities, pd.DataFrame):
            try:
                # If it's a path to a CSV, load it
                if isinstance(df_identities, str) and os.path.exists(df_identities):
                    df_identities = pd.read_csv(df_identities)
                else: # Try to convert if it's list of dicts or similar
                    df_identities = pd.DataFrame(df_identities)
            except Exception as e:
                 return f"Error: Could not process identities data: {e}"


        row_series = df_identities[df_identities['Clone Number'] == clone_number]
        if row_series.empty:
            return f"Error: Could not find data for selected identity '{selected_identity}'."
        row = row_series.iloc[0]

        if not caption:
            caption = suggest_caption(row)
            print(f"Using suggested caption: {caption}")

        consumer_key = os.getenv("X_CONSUMER_KEY")
        consumer_secret = os.getenv("X_CONSUMER_SECRET")
        access_token = os.getenv("X_ACCESS_TOKEN")
        access_token_secret = os.getenv("X_ACCESS_TOKEN_SECRET")

        if not all([consumer_key, consumer_secret, access_token, access_token_secret]):
            return "Error: X API credentials not found in environment variables. (X_CONSUMER_KEY, X_CONSUMER_SECRET, X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET)"

        client = tweepy.Client(
            consumer_key=consumer_key, consumer_secret=consumer_secret,
            access_token=access_token, access_token_secret=access_token_secret
        )
        
        # For media upload, tweepy.API is still often used or a v1.1 endpoint via client
        auth_v1 = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
        api_v1 = tweepy.API(auth_v1)
        
        media = api_v1.media_upload(filename=image_path)
        client.create_tweet(text=caption, media_ids=[media.media_id_string]) # Use media_id_string

        return f"Successfully shared to X: {caption}"
    except tweepy.TweepyException as e:
        error_msg = f"Twitter API error: {str(e)}"
        print(error_msg)
        with open('error_log.txt', 'a') as f: f.write(f"{datetime.now()}: {error_msg}\n")
        return error_msg

    except Exception as e:
        error_msg = f"Error sharing to X: {str(e)} (Type: {type(e).__name__})"
        print(error_msg)
        import traceback
        traceback.print_exc()
        with open('error_log.txt', 'a') as f: f.write(f"{datetime.now()}: {error_msg}\nTrace: {traceback.format_exc()}\n")
        return error_msg

# --- End of Section 4 ---
# --- Start of Section 5 ---

def generate_energy_signature(row):
    adjectives = ['Fiery', 'Ethereal', 'Sizzling', 'Soulful', 'Insightful', 'Electric', 'Vibrant', 'Quantum', 'Nebula']
    elements = ['Cosmic Blaze', 'Starlight', 'Cosmic Fizzle', 'Pulse', 'Ocean Whisper', 'Sky Breeze', 'Moon Glow', 'Heartbeat']
    
    # Influence by profession
    profession = row.get('Profession', 'Unknown')
    if profession in ['Hacker', 'DJ', 'Streamer']:
        adjectives = ['Electric', 'Vibrant', 'Sizzling'] + adjectives
    elif profession in ['Astrologer', 'High Priestess Witch']:
        adjectives = ['Ethereal', 'Quantum', 'Nebula'] + adjectives
    
    # Influence by hair color
    hair_color = row.get('Hair color', 'Unknown')
    if 'Blue' in hair_color or 'Purple' in hair_color:
        elements = ['Starlight', 'Moon Glow'] + elements
    elif 'Red' in hair_color or 'Orange' in hair_color:
        elements = ['Cosmic Blaze', 'Cosmic Fizzle'] + elements
    
    return f"{random.choice(adjectives)} {random.choice(elements)}"

# Generate identities

def generate_identities_gui(num_identities, resume_training, profession_filter, playlist_filter, le_dict_param, scaler_age_param, scaler_height_param, scaler_weight_param, scaler_measurements_param, scaler_features_param, df_param, first_names_list, last_names_list, nicknames_list, first_name_gen_model, last_name_gen_model, nickname_gen_model, additional_names_df):
    global model # Use the globally defined model
    
    # Use parameters instead of global variables for clarity and testability
    # le_dict, scaler_age, etc. are now passed as params.

    if resume_training and os.path.exists('model.pth'):
        try:
            model.load_state_dict(torch.load('model.pth', map_location=device))
            model.to(device) # Ensure model is on correct device
            print("Resumed training from model.pth")
        except Exception as e:
            print(f"Could not load model.pth for resuming: {e}. Training from scratch.")
            # Fall through to train from scratch if loading fails
    
    generated_firstnames_set = set(additional_names_df['Firstname'].dropna().tolist())
    generated_lastnames_set = set(additional_names_df['Lastname'].dropna().tolist())
    # Assuming nicknames are not saved back to previous_names.csv in the same way, start fresh or load if available
    generated_nicknames_set = set() 
    
    identities = []
    training_losses = [] # Renamed from losses
    total_epochs_trained = 0 # Renamed from total_epochs
    cycles_train = 5
    epochs_per_cycle_train = 20
    
    # Use the globally defined features array for training the main IdentityGenerator model
    # Ensure 'features' is defined in the global scope and is the correct data
    global features # Explicitly state usage of global features for main model training
    if 'features' not in globals() or not isinstance(features, np.ndarray):
        # This is a fallback, ideally features should be prepared correctly before this call
        print("Error: Global 'features' array for model training is not properly defined. Aborting generation.")
        # Yield an error state for Gradio
        fig_err, ax_err = plt.subplots()
        ax_err.text(0.5, 0.5, "Error: Training data (features) missing.", ha='center', va='center')
        yield pd.DataFrame(), None, "error_plot.png", gr.update(choices=["None"]), None, 0, "Error: Training data missing.", fig_err
        plt.close(fig_err)
        return


    for cycle_run_losses, cycle_run_epochs in train_model(model, features, cycles=cycles_train, epochs_per_cycle=epochs_per_cycle_train, verbose=False):
        training_losses.extend(cycle_run_losses) # Accumulate losses from each yield
        total_epochs_trained = cycle_run_epochs # Update total epochs from yield

        current_cycle_display = min((total_epochs_trained -1) // epochs_per_cycle_train + 1 if total_epochs_trained > 0 else 1, cycles_train)
        current_epoch_display = (total_epochs_trained -1) % epochs_per_cycle_train + 1 if total_epochs_trained > 0 else 0
        progress_val = min((total_epochs_trained / (cycles_train * epochs_per_cycle_train)) * 100, 100) if (cycles_train * epochs_per_cycle_train > 0) else 0
        
        try:
            fig, ax = plt.subplots()
            fig.patch.set_alpha(0) # Transparent background for plot
            ax.set_facecolor('#0a0a28') # Dark background for axes
            ax.plot(training_losses, color='#00e6e6', linewidth=2, label='Loss')
            ax.set_title('Training Loss', color='#00ffcc', fontsize=14, pad=15)
            ax.set_xlabel('Epoch', color='#00ffcc', fontsize=12)
            ax.set_ylabel('Loss', color='#00ffcc', fontsize=12)
            ax.tick_params(axis='both', colors='#00e6e6')
            ax.grid(True, color='#00e6e6', alpha=0.3, linestyle='--')
            for spine in ax.spines.values(): spine.set_color('#00e6e6')

            # Yield the results without the figure
            # Save plot for download button after yielding
            try:
                fig.savefig("loss_plot.png", facecolor=fig.get_facecolor())
                plot_path = os.path.abspath("loss_plot.png")
            except Exception as e_save:
                print(f"Error saving loss plot: {e_save}")
                plot_path = None
                print(f"DEBUG: Yield 1 - Dataframe: None, CSV: None, Plot: {plot_path}, Dropdown: ['None'], Progress: {progress_val}, Status: 'Training: Cycle {current_cycle_display}/{cycles_train}, Epoch {current_epoch_display}/{epochs_per_cycle_train}'")
                yield None, None, plot_path, gr.update(choices=["None"]), None, progress_val, f"Training: Cycle {current_cycle_display}/{cycles_train}, Epoch {current_epoch_display}/{epochs_per_cycle_train}"
        
        except Exception as e_fig:
            print(f"Error with plot operations: {e_fig}")
            # Yield without the figure or plot file if there was an error
            print(f"DEBUG: Yield 2 - Dataframe: None, CSV: None, Plot: None, Dropdown: ['None'], Progress: {progress_val}, Status: 'Training: Cycle {current_cycle_display}/{cycles_train}, Epoch {current_epoch_display}/{epochs_per_cycle_train}'")
            yield None, None, None, gr.update(choices=["None"]), None, progress_val, f"Training: Cycle {current_cycle_display}/{cycles_train}, Epoch {current_epoch_display}/{epochs_per_cycle_train}"
        
        finally:
            plt.close(fig)  # Ensure figure is closed even if there was an error
        time.sleep(0.1) # Small delay for UI to update

    try:
        torch.save(model.state_dict(), 'model.pth')
        print(f"Main IdentityGenerator model saved to model.pth")
    except Exception as e_save:
        print(f"Error saving main model: {e_save}")

    final_loss_str = f"{training_losses[-1]:.6f}" if training_losses else "N/A"
    print(f"Training Summary: Total Epochs: {total_epochs_trained}, Final Loss: {final_loss_str}")
    
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        firstnames_batch = []
        lastnames_batch = []
        nicknames_batch_gen = [] # Renamed to avoid conflict

        batch_size_names = 32 
        for _ in range(0, num_identities): # Generate enough names for num_identities
            # First Name
            firstname = generate_name(first_name_gen_model, first_name_char_to_idx, first_name_idx_to_char,
                                   first_name_max_len, device, name_type='firstname',
                                   existing_names=generated_firstnames_set, temperature=0.7)
            firstnames_batch.append(firstname)
            generated_firstnames_set.add(firstname)

            # Last Name
            lastname = generate_name(last_name_gen_model, last_name_char_to_idx, last_name_idx_to_char,
                                  last_name_max_len, device, name_type='lastname',
                                  existing_names=generated_lastnames_set, temperature=0.7)
            lastnames_batch.append(lastname)
            generated_lastnames_set.add(lastname)

            # Nickname
            nickname = generate_name(nickname_gen_model, nickname_char_to_idx, nickname_idx_to_char,
                                  nickname_max_len, device, name_type='nickname',
                                  existing_names=generated_nicknames_set, temperature=0.7)
            nicknames_batch_gen.append(nickname)
            generated_nicknames_set.add(nickname)


    # Initialize predefined playlists once
    playlists = [f"{artist}'s {genre} {theme} Mix"
                 for artist in artist_names_list
                 for genre in genres_list
                 for theme in themes_list]
# Define female names (subset; expand if first_names_list has gender tags)

    for i in range(num_identities):
        female_names = ['Jeanette', 'Violet', 'Xena', 'Ximena', 'Tara', 'Ceamelly', 'Ella', 'Leila', 'Ellie', 'Sophia', 'Nia', 'Zoe', 'Aria', 'Luna', 'Emma']
        firstname = female_names[i % len(female_names)] if i < len(firstnames_batch) else "FemaleFallback"
        lastname = lastnames_batch[i] if i < len(lastnames_batch) else "LastFallback"
        nickname = nicknames_batch_gen[i] if i < len(nicknames_batch_gen) else "NickFallback"
        identity = {
             'Clone Number': f'CLN-{i+1:03d}',
             'Firstname': firstname,
             'Lastname': lastname,
             'Nickname': nickname,
             'Gender': 'Female',
             'Profession': np.random.choice(list(le_dict_param['Profession'].classes_)) if profession_filter == "All" else profession_filter,
             'Playlist': np.random.choice(playlists) if playlist_filter == "All" else playlist_filter
             }
        print(f"DEBUG: Generating identity {i+1}/{num_identities}: {identity}")
        
        # Sample from the original df_param for base features, then transform and inverse_transform
        if df_param.empty or not all(col in df_param.columns for col in features_cols):
            print(f"Error: df_param is empty or missing required feature columns for identity {i+1}. Skipping.")
            identity['Clone Number'] = f'ERR-{i+1:03d}'
            identity['Error'] = 'Attribute generation failed'
            identities.append(identity)
            print(f"DEBUG: Appended error identity {i+1}: {identity['Clone Number']}")
            continue
            
        # If df_param is valid, proceed with sampling and transformations
            sampled_row_features = df_param.sample(1)[features_cols].values
            input_features_tensor = torch.tensor(scaler_features_param.transform(sampled_row_features), dtype=torch.float32).to(device)

            hidden, cell = model.init_hidden(1)
            output_tensor, _, _ = model(input_features_tensor, hidden, cell)
            output_scaled = output_tensor.cpu().detach().numpy().squeeze(0)
            if output_scaled.ndim == 1:
                output_scaled = output_scaled.reshape(1, -1)

            output_inversed = scaler_features_param.inverse_transform(output_scaled)

            age = int(scaler_age_param.inverse_transform([[output_inversed[0, features_cols.index('Age')]]])[0, 0])
            height = int(scaler_height_param.inverse_transform([[output_inversed[0, features_cols.index('Height')]]])[0, 0])
            weight = int(scaler_weight_param.inverse_transform([[output_inversed[0, features_cols.index('Weight')]]])[0, 0])

            bust_val = output_inversed[0, features_cols.index('Bust')]
            waist_val = output_inversed[0, features_cols.index('Waist')]
            hips_val = output_inversed[0, features_cols.index('Hips')]

            measurements_scaled_for_inv = np.array([[bust_val, waist_val, hips_val]])
            measurements_inversed = scaler_measurements_param.inverse_transform(measurements_scaled_for_inv)
            bust, waist, hips = int(measurements_inversed[0, 0]), int(measurements_inversed[0, 1]), int(measurements_inversed[0, 2])

            def safe_inverse_transform(le, val_idx, default_val="Unknown"):
                try:
                    int_val = int(round(output_inversed[0, val_idx]))
                    if 0 <= int_val < len(le.classes_):
                        return le.inverse_transform([int_val])[0]
                    else:
                        return random.choice(le.classes_) if len(le.classes_) > 0 else default_val
                except Exception:
                    return random.choice(le.classes_) if len(le.classes_) > 0 else default_val

            nationality = safe_inverse_transform(le_dict_param['Nationality'], features_cols.index('Nationality'))
            ethnicity = safe_inverse_transform(le_dict_param['Ethnicity'], features_cols.index('Ethnicity'))
            birthplace = safe_inverse_transform(le_dict_param['Birthplace'], features_cols.index('Birthplace'))
            profession = safe_inverse_transform(le_dict_param['Profession'], features_cols.index('Profession'))
            body_type = safe_inverse_transform(le_dict_param['Body type'], features_cols.index('Body type'))
            hair_color = safe_inverse_transform(le_dict_param['Hair color'], features_cols.index('Hair color'))
            eye_color = safe_inverse_transform(le_dict_param['Eye color'], features_cols.index('Eye color'))
            bra_size = safe_inverse_transform(le_dict_param['Bra/cup size'], features_cols.index('Bra/cup size'))
            boobs = safe_inverse_transform(le_dict_param['Boobs'], features_cols.index('Boobs'))

            born = (datetime.now() - timedelta(days=int(age * 365.25))).strftime('%Y-%m-%d')
            body_measurements_str = f"{bust}-{waist}-{hips}"

            sister_of = 'None'
            if random.random() < 0.1 and identities:
                sister_of = random.choice(identities)['Clone Number']

            energy_signature = generate_energy_signature(identity)

            cosmic_tattoo = 'None'
            if random.random() < 0.08:
                tattoo_options = ['Starfield Nebula', 'Galactic Spiral', 'Pulsar Wave', 'Celestial Serpent', 'Lunar Crest', 'Eclipse Rune']
                cosmic_tattoo = random.choice(tattoo_options)

            cosmic_playlist = 'None'
            if random.random() < 0.08:
                cosmic_playlist = f"{random.choice(artist_names_list)}'s {random.choice(genres_list)} {random.choice(themes_list)} Mix"

            cosmic_pet = 'None'
            if random.random() < 0.05:
                pet_options = ['Nebula Kitten', 'Pulsar Pup', 'Quantum Finch', 'Melody Finch named Harmony', 'Stardust Bunny', 'Aether Fox']
                cosmic_pet = random.choice(pet_options)
                if cosmic_pet == 'Melody Finch named Harmony' and random.random() < 0.5:
                    cosmic_pet = f"{cosmic_pet} ({random.choice(['inspiring', 'uplifting'])})"

            cosmic_artifact = 'None'
            if random.random() < 0.03:
                artifact_options = ['Quantum Locket', 'Stellar Compass', 'Nebula Orb', 'Starforge Hammer', 'Astral Lantern', 'Void Crystal']
                cosmic_artifact = random.choice(artifact_options)

            cosmic_aura = 'None'
            if random.random() < 0.03:
                aura_options = ['Aurora Veil', 'Stellar Mist', 'Pulsar Halo', 'Ethereal Glow', 'Nebula Shimmer', 'Quantum Spark']
                cosmic_aura = random.choice(aura_options)

            cosmic_hobby = 'None'
            if random.random() < 0.05:
                hobby_options = ['Nebula Painting', 'Quantum Dance', 'Starlight Poetry', 'Astral Weaving', 'Cosmic Sculpting', 'Galactic Songwriting']
                cosmic_hobby = random.choice(hobby_options)

            cosmic_destiny = 'None'
            if random.random() < 0.05:
                destiny_options = ['Nebula Voyager', 'Pulsar Poet', 'Quantum Pathfinder', 'Galactic Harmonizer', 'Starlight Seer', 'Aether Guardian']
                cosmic_destiny = random.choice(destiny_options)

            cosmic_ability = 'None'
            if random.random() < 0.03:
                ability_options = ['Starweaving', 'Nebula Cloaking', 'Quantum Leap', 'Astral Projection', 'Pulsar Blast', 'Ethereal Song']
                cosmic_ability = random.choice(ability_options)

            identity.update({
                'Gender': 'Female',
                'Age': age,
                'Born': born_date.strftime('%Y-%m-%d'),
                'Nationality': nationality,
                'Ethnicity': ethnicity,
                'Birthplace': birthplace,
                'Profession': profession,
                'Height': height,
                'Weight': weight,
                'Body type': np.random.choice(list(le_dict_param['Body type'].classes_)),
                'Body Measurements': body_measurements_str,
                'Hair color': hair_color,
                'Eye color': eye_color,
                'Bra/cup size': bra_size,
                'Boobs': boobs,
                'Sister Of': sister_of,
                'Energy Signature': energy_signature,
                'Cosmic Tattoo': cosmic_tattoo,
                'Cosmic Playlist': cosmic_playlist,
                'Cosmic Pet': cosmic_pet,
                'Cosmic Artifact': cosmic_artifact,
                'Cosmic Aura': cosmic_aura,
                'Cosmic Hobby': cosmic_hobby,
                'Cosmic Destiny': cosmic_destiny,
                'Cosmic Ability': cosmic_ability
            })

            identities.append(identity)
            print(f"DEBUG: Appended identity {i+1}: {identity['Clone Number']}")
            df_identities_current = pd.DataFrame(identities, columns=[
                'Clone Number', 'Firstname', 'Lastname', 'Nickname', 'Gender', 'Profession', 'Playlist',
                'Age', 'Born', 'Nationality', 'Ethnicity', 'Birthplace', 'Height', 'Weight', 'Body type',
                'Body Measurements', 'Hair color', 'Eye color', 'Bra/cup size', 'Boobs', 'Sister Of',
                'Energy Signature', 'Cosmic Tattoo', 'Cosmic Playlist', 'Cosmic Pet', 'Cosmic Artifact',
                'Cosmic Aura', 'Cosmic Hobby', 'Cosmic Destiny', 'Cosmic Ability'
            ])
            # Apply filters
            df_filtered = df_identities_current.copy() # Work on a copy for filtering
            if profession_filter != 'All':
                df_filtered = df_filtered[df_filtered['Profession'] == profession_filter]
            if playlist_filter != 'All': # Filter by Cosmic Playlist Theme
                df_filtered = df_filtered[df_filtered['Cosmic Playlist'].str.contains(f"{playlist_filter}", case=False, na=False)]

            # Save all generated identities (unfiltered) to CSV
            generated_csv_path = 'generated_cha_identities.csv'
            try:
                df_identities_current.to_csv(generated_csv_path, index=False)
            except PermissionError:
                print(f"Error: Permission denied writing to {generated_csv_path}. Check file permissions.")
                generated_csv_path = None # Indicate failure
            except Exception as e_csv:
                print(f"Error writing to {generated_csv_path}: {e_csv}")
                generated_csv_path = None

            # Update previous_names.csv
            # Only add if the name generation was successful for this identity
            if firstname != "FirstFallback" and lastname != "LastFallback":
                new_name_df = pd.DataFrame([{'Firstname': firstname, 'Lastname': lastname}])
                # Concatenate carefully, avoid issues if additional_names_df is None or not DataFrame
                if isinstance(additional_names_df, pd.DataFrame):
                    additional_names_df = pd.concat([additional_names_df, new_name_df], ignore_index=True)
                else: # Initialize if it wasn't a DataFrame
                    additional_names_df = new_name_df
                try:
                    additional_names_df.to_csv('previous_names.csv', index=False)
                except PermissionError: print("Error: Permission denied writing to previous_names.csv.")
                except Exception as e_prev_csv: print(f"Error writing to previous_names.csv: {e_prev_csv}")
            
            # Create identity list for dropdown from the *filtered* DataFrame
            identity_list_dropdown = ["None"] + [f"{row_filt['Clone Number']}: {row_filt['Nickname']}" for _, row_filt in df_filtered.iterrows()]
            
            # Create an empty figure for non-training updates
            empty_fig, empty_ax = plt.subplots()
            empty_ax.set_facecolor('#0a0a28')
            empty_fig.patch.set_alpha(0)
            empty_ax.text(0.5, 0.5, "Generation in progress...", ha='center', va='center', color='#00ffcc')
            empty_ax.set_xticks([])
            empty_ax.set_yticks([])
            for spine in empty_ax.spines.values():
                spine.set_color('#00e6e6')

            # Save current plot state for download
            if os.path.exists("loss_plot.png"):
                current_plot = "loss_plot.png"
            else:
                empty_fig.savefig("generation_progress.png", facecolor=empty_fig.get_facecolor())
                current_plot = "generation_progress.png"

            # Yield the *filtered* DataFrame for display without the figure
            csv_path = os.path.abspath(generated_csv_path) if generated_csv_path and os.path.exists(generated_csv_path) else None
            plot_path = os.path.abspath(current_plot) if current_plot and os.path.exists(current_plot) else None
            print(f"DEBUG: Yield 3 - Dataframe: {df_filtered.shape if df_filtered is not None else None}, CSV: {csv_path}, Plot: {plot_path}, Dropdown: {identity_list_dropdown}, Progress: {((i+1)/num_identities)*100}, Status: 'Generated {i+1}/{num_identities} identities'")
            yield df_filtered, csv_path, plot_path, gr.update(choices=identity_list_dropdown), None, ((i+1)/num_identities)*100, f"Generated {i+1}/{num_identities} identities"
            plt.close(empty_fig)  # Clean up the figure
            time.sleep(0.05) # Shorter sleep during generation phase

    # Final yield after loop
    df_final_identities = pd.DataFrame(identities)
    df_final_filtered = df_final_identities.copy()
    if profession_filter != 'All': df_final_filtered = df_final_filtered[df_final_filtered['Profession'] == profession_filter]
    if playlist_filter != 'All': df_final_filtered = df_final_filtered[df_final_filtered['Cosmic Playlist'].str.contains(f"{playlist_filter}", case=False, na=False)]
    
    final_identity_list_dropdown = ["None"] + [f"{row_fin['Clone Number']}: {row_fin['Nickname']}" for _, row_fin in df_final_filtered.iterrows()]
    final_loss_plot_path = "loss_plot.png" if os.path.exists("loss_plot.png") else None
    final_csv_path = 'generated_cha_identities.csv' if os.path.exists('generated_cha_identities.csv') else None

    # Create final plot
    final_fig, final_ax = plt.subplots()
    final_ax.set_facecolor('#0a0a28')
    final_fig.patch.set_alpha(0)
    if training_losses:
        final_ax.plot(training_losses, color='#00e6e6', linewidth=2, label='Loss')
        final_ax.set_title('Final Training Loss', color='#00ffcc', fontsize=14, pad=15)
        final_ax.set_xlabel('Epoch', color='#00ffcc', fontsize=12)
        final_ax.set_ylabel('Loss', color='#00ffcc', fontsize=12)
        final_ax.tick_params(axis='both', colors='#00e6e6')
        final_ax.grid(True, color='#00e6e6', alpha=0.3, linestyle='--')
    else:
        final_ax.text(0.5, 0.5, "Generation Complete!", ha='center', va='center', color='#00ffcc')
        final_ax.set_xticks([])
        final_ax.set_yticks([])
    for spine in final_ax.spines.values():
        spine.set_color('#00e6e6')

    # Save final plot for download
    try:
        final_fig.savefig("loss_plot.png", facecolor=final_fig.get_facecolor())
        final_loss_plot_path = "loss_plot.png"
    except Exception as e:
        print(f"Error saving final plot: {e}")
        final_loss_plot_path = None

    csv_path = os.path.abspath(final_csv_path) if final_csv_path and os.path.exists(final_csv_path) else None
    plot_path = os.path.abspath(final_loss_plot_path) if final_loss_plot_path and os.path.exists(final_loss_plot_path) else None
    print(f"DEBUG: Yield 4 - Dataframe: {df_final_filtered.shape if df_final_filtered is not None else None}, CSV: {csv_path}, Plot: {plot_path}, Dropdown: {final_identity_list_dropdown}, Progress: 100, Status: 'Generation Complete!'")
    yield df_final_filtered, csv_path, plot_path, gr.update(choices=final_identity_list_dropdown), None, 100, "Generation Complete!"
    plt.close(final_fig)  # Clean up the figure


def generate_identities_gui_wrapper(num_identities, resume_training, profession_filter, playlist_filter):
    print("Available professions in dropdown:", list(le_dict['Profession'].classes_)) # le_dict should be globally accessible here
    # Pass all required data to the main generation function
    # Ensure all these variables are correctly defined in the global scope or passed appropriately
    # df, first_names, last_names, nicknames are from global scope
    # *_gen models are also from global scope now after load_or_train
    # additional_names is also global
    
    # Make sure these are the correct, updated instances after any loading/training
    global first_name_gen, last_name_gen, nickname_gen, additional_names
    
    for result in generate_identities_gui(
        num_identities, resume_training, profession_filter, playlist_filter,
        le_dict, scaler_age, scaler_height, scaler_weight, scaler_measurements, scaler_features,
        df, first_names, last_names, nicknames,
        first_name_gen, last_name_gen, nickname_gen,
        additional_names # Pass the global DataFrame
    ):
        yield result


# --- End of Section 5 ---
# --- Start of Section 6 ---

# CSS
custom_css = """
body { /* ... existing CSS ... */ }
/* Add any new CSS or keep existing */
"""
# Your existing CSS remains here. For brevity, I'm not repeating it.
# Ensure your full CSS is included when you save the file.
custom_css = """
body {
    background: transparent;
    color: #00e6e6;
}
.gradio-container {
    max-width: 3200px;
    margin: auto;
    border: 2px solid #00e6e6;
    border-radius: 15px;
    background: rgba(10, 10, 40, 0.8);
    padding: 20px;
}
h1 {
    text-align: center;
    color: #00ffcc;
    text-shadow: 0 0 15px #00ffcc;
}
h2, h3 {
    text-align: center;
    color: #00ffcc;
}
button {
    background: #1a1a4d;
    color: #00e6e6;
    border: 2px solid #00e6e6;
    border-radius: 10px;
    padding: 10px 20px;
    transition: all 0.3s ease;
}
button:hover {
    background: #00e6e6;
    color: #0d0d2b;
    box-shadow: 0 0 25px #00e6e6;
    transform: scale(1.1);
}
.dataframe-container {
    width: 100% !important;
    overflow-x: auto;
    background: rgba(20, 20, 60, 0.9);
    border: 1px solid #00e6e6;
    border-radius: 10px;
    padding: 10px;
}
.dataframe table {
    width: auto;
    min-width: 100%;
    border-collapse: collapse;
    font-size: 14px;
}
.dataframe th, .dataframe td {
    padding: 6px;
    text-align: left;
    border: 1px solid #00e6e6;
    white-space: nowrap;
    max-width: 1200px; /* Increased max-width for better readability */
    min-width: 50px;
    overflow: hidden;
    text-overflow: ellipsis;
    position: relative;
}
.dataframe th {
    background: rgba(0, 230, 230, 0.1);
    position: sticky;
    top: 0;
    z-index: 10;
}
/* ... rest of your original CSS ... */
.dataframe-container::-webkit-scrollbar { height: 8px; }
.dataframe-container::-webkit-scrollbar-track { background: #0a0a28; }
.dataframe-container::-webkit-scrollbar-thumb { background: #00e6e6; border-radius: 4px; }
.dataframe-container::-webkit-scrollbar-thumb:hover { background: #00ffcc; }
.dataframe tr:has(td:last-child:not(:contains("None"))) { background: rgba(0, 255, 255, 0.2) !important; }
.dataframe tr:has(td:contains("Fiery")) { box-shadow: 0 0 10px rgba(255, 100, 100, 0.5) !important; }
.dataframe tr:has(td:contains("Quantum Pathfinder")):hover::after {
    content: "A seeker of quantum pathways"; position: absolute; background: #0a0a28; color: #00ffcc;
    padding: 5px; border: 1px solid #00e6e6; border-radius: 5px; z-index: 100;
    top: -30px; left: 50%; transform: translateX(-50%); white-space: nowrap;
}
/* Add other specific keyword highlights and animations as you had */
@keyframes pulse {
    0% { box-shadow: 0 0 5px rgba(50, 200, 255, 0.7); }
    50% { box-shadow: 0 0 15px rgba(50, 200, 255, 1); }
    100% { box-shadow: 0 0 5px rgba(50, 200, 255, 0.7); }
}
.dataframe tr:has(td:contains("Starweaving")), .dataframe tr:has(td:contains("Nebula Cloaking")),
.dataframe tr:has(td:contains("Quantum Leap")), .dataframe tr:has(td:contains("Astral Projection")),
.dataframe tr:has(td:contains("Pulsar Blast")), .dataframe tr:has(td:contains("Ethereal Song")),
.dataframe tr:has(td:contains("Electric")), .dataframe tr:has(td:contains("Vibrant")),
.dataframe tr:has(td:contains("Nebula")), .dataframe tr:has(td:contains("Quantum Locket")),
.dataframe tr:has(td:contains("Stellar Compass")), .dataframe tr:has(td:contains("Nebula Orb")),
.dataframe tr:has(td:contains("Nebula Kitten")), .dataframe tr:has(td:contains("Pulsar Pup")),
.dataframe tr:has(td:contains("Quantum Finch")), .dataframe tr:has(td:contains("Aurora Veil")),
.dataframe tr:has(td:contains("Stellar Mist")), .dataframe tr:has(td:contains("Pulsar Halo")),
.dataframe tr:has(td:contains("Nebula Painting")), .dataframe tr:has(td:contains("Quantum Dance")),
.dataframe tr:has(td:contains("Starlight Poetry")) {
    animation: pulse 1.5s infinite;
}
/* Add specific box-shadows for keywords as you had them, e.g.: */
.dataframe tr:has(td:contains("Ethereal")) { box-shadow: 0 0 10px rgba(100, 255, 255, 0.5) !important; }
.dataframe tr:has(td:contains("Sizzling")) { box-shadow: 0 0 10px rgba(255, 50, 50, 0.7) !important; }
/* ... and so on for all your keyword-based stylings and hover effects ... */
"""


# Gradio Interface
with gr.Blocks(css=custom_css, theme=gr.themes.Base(primary_hue=gr.themes.colors.cyan, secondary_hue=gr.themes.colors.blue)) as demo: # Slightly updated theme
    gr.Markdown("# Neural Identity Matrix V24.35")

    with gr.Tab("Generate Identities"):
        with gr.Row():
            profession_filter_dd = gr.Dropdown( # Renamed to avoid conflict if any global with same name
                choices=['All'] + (list(le_dict['Profession'].classes_) if 'Profession' in le_dict else ['All', 'Error: Professions not loaded']),
                value='All',
                label="Filter by Profession"
            )
            playlist_filter_dd = gr.Dropdown( # Renamed
                choices=['All'] + (themes_list if themes_list else ['All', 'Error: Themes not loaded']),
                value='All',
                label="Filter by Cosmic Playlist Theme"
            )
        with gr.Row():
            num_identities_slider = gr.Slider(minimum=1, maximum=250, value=25, step=1, label="Number of Identities to Generate") # Renamed
            resume_training_cb = gr.Checkbox(label="Resume Main Model Training from Checkpoint (model.pth)", value=False) # Renamed, clarified
        
        generate_button = gr.Button("Initialize Identity Generation & Train/Load Models")
        
        with gr.Row():
            style_dropdown = gr.Dropdown(label="Image Style", choices=["Cyberpunk", "Real-Life", "Instagram Selfie", "Cinematic"], value="Cyberpunk")

        output_df = gr.Dataframe(
            label="Generated Identities",
            wrap=True,
            show_copy_button=True,
            interactive=False,
            elem_classes="full-width-dataframe",
            )
        with gr.Row():
                 with gr.Column(scale=1, min_width=300):
                     csv_download = gr.File(label="Download Generated Identities (CSV)", value="generated_cha_identities.csv")
                 with gr.Column(scale=1, min_width=300):
                    plot_download = gr.File(label="Download Loss Plot (PNG)", value="loss_plot.png")  # Fixed incorrect value
                 with gr.Column(scale=2, min_width=600):
                    loss_plot = gr.Plot(label="Training Loss Over Epochs")

        with gr.Row():
                identity_dropdown_select = gr.Dropdown(choices=["None"], label="Select Identity for Image/Audio/Share")
                progress_bar_identities = gr.Slider(minimum=0, maximum=100, value=0, label="Identity Generation Progress", interactive=False)
                status_message_identities = gr.Textbox(label="Identity Generation Status", lines=2)
                image_output_display = gr.Image(label="Generated Image (from Identity Tab - not primary)")  # Still commented out
    with gr.Tab("Generate Images"):
        with gr.Row():
            allow_nsfw_cb = gr.Checkbox(label="Allow NSFW Content", value=False) # Renamed
            style_theme_dd = gr.Dropdown(choices=style_themes_list if style_themes_list else ["Cyberpunk"], value="Cyberpunk", label="Style Theme") # Renamed
            location_dd = gr.Dropdown(choices=locations_list if locations_list else ["Cosmic Nebula"], value="Cosmic Nebula", label="Location") # Renamed
            overall_theme_dd = gr.Dropdown(choices=overall_themes_list if overall_themes_list else ["Ethereal Dreamscape"], value="Ethereal Dreamscape", label="Overall Theme") # Renamed
        # --- MODIFICATION START ---
        image_seed_input = gr.Number(label="Image Seed (0 for random)", value=0, precision=0, minimum=0) # Added seed input
        # --- MODIFICATION END ---
        
        generate_image_button = gr.Button("Generate Image for Selected Identity")
        batch_generate_button = gr.Button("Generate Images for All Displayed Identities (Batch)") # Clarified target
        
        with gr.Row():
            image_output_main = gr.Image(label="Generated Image", type="filepath") # Renamed, type filepath
            gallery_output_display = gr.Gallery(label="Image Gallery (from generated_images folder)", columns=5, height="auto") # Renamed

        batch_status_text = gr.Textbox(label="Batch Generation Status", lines=2) # Renamed
        batch_progress_slider = gr.Slider(minimum=0, maximum=100, value=0, label="Batch Progress", interactive=False) # Renamed
    with gr.Tab("Generate Audio Prompt"):
        generate_audio_button = gr.Button("Generate Audio Prompt for Selected Identity")
        audio_prompt_output_text = gr.Textbox(label="Audio Prompt", lines=5) # Renamed
    with gr.Tab("Share to X"):
        caption_input_text = gr.Textbox(label="Caption (leave blank for suggestion)", placeholder="Enter a caption or leave blank for a suggested one", lines=3) # Renamed
        share_button_x = gr.Button("Share Selected Image to X") # Renamed
        share_status_text = gr.Textbox(label="Share Status", lines=2) # Renamed
    # Wire components
    generate_button.click(
        fn=generate_identities_gui_wrapper,
        inputs=[num_identities_slider, resume_training_cb, profession_filter_dd, playlist_filter_dd],
        outputs=[
            output_df,  # DataFrame display
            csv_download,  # CSV file output
            plot_download,   # Plot PNG file output
            identity_dropdown_select,  # Identity dropdown
            progress_bar_identities,  # Progress bar
            status_message_identities,  # Status message
            loss_plot  # Loss plot display
        ],
        queue=True
    )

    generate_image_button.click(
        fn=generate_flux_image,
        # --- MODIFICATION START ---
        inputs=[identity_dropdown_select, output_df, allow_nsfw_cb, style_theme_dd, location_dd, overall_theme_dd, image_seed_input],
        # --- MODIFICATION END ---
        outputs=[image_output_main, status_message_identities], # Status message from identity tab might be confusing, consider separate status for image gen
        queue=True
    )

    batch_generate_button.click(
        fn=generate_images_batch,
        inputs=[output_df, gr.State(value=5), allow_nsfw_cb, style_theme_dd, location_dd, overall_theme_dd], # Smaller default batch for UI responsiveness
        outputs=[image_output_main, batch_status_text, gallery_output_display, batch_progress_slider],
        queue=True
    )
    
    # Action to refresh gallery when image is generated by single button too
    # This can be done by returning an updated gallery from generate_flux_image or adding a refresh button
    # For now, let's make the batch button also update the gallery. Single image could too.
    def update_gallery_after_single_image(image_path, status_msg): # Wrapper to also refresh gallery
        try:
            # Update gallery
            gallery_items = display_image_gallery(None)
            
            # Handle plot file
            if os.path.exists("loss_plot.png"):
                current_plot = "loss_plot.png"
            else:
                # Create an empty plot if none exists
                fig, ax = plt.subplots()
                ax.set_facecolor('#0a0a28')
                fig.patch.set_alpha(0)
                ax.text(0.5, 0.5, "Gallery Updated", ha='center', va='center', color='#00ffcc')
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color('#00e6e6')
                try:
                    fig.savefig("gallery_update.png", facecolor=fig.get_facecolor())
                    current_plot = "gallery_update.png"
                finally:
                    plt.close(fig)
            
            # Return all required outputs
            return image_path, status_msg, gallery_items, current_plot
        except Exception as e:
            print(f"Error in gallery update: {e}")
            return image_path, f"Gallery update error: {str(e)}", None, None

    generate_image_button.click( # Second click action for gallery update
        fn=update_gallery_after_single_image,
        inputs=[image_output_main, status_message_identities], # Takes output of first click
        outputs=[image_output_main, status_message_identities, gallery_output_display, plot_download], # Updates gallery, plot, and download button
        queue=True # Ensure it runs after the image is generated
    )


    generate_audio_button.click(
        fn=generate_audio_prompt,
        inputs=[identity_dropdown_select, output_df, style_theme_dd, location_dd, overall_theme_dd],
        outputs=[audio_prompt_output_text],
        queue=True
    )

    share_button_x.click(
        fn=share_to_x,
        inputs=[image_output_main, caption_input_text, output_df, identity_dropdown_select],
        outputs=[share_status_text],
        queue=True
    )

# --- End of Section 6 ---
# --- Start of Section 7 ---

def main():
    global df, features, first_name_gen, last_name_gen, nickname_gen, additional_names
    if verify_required_files():
        demo.launch()

# Launch the interface
if __name__ == "__main__": # Standard Python practice
    main()

# --- End of Section 7 ---
# --- End of Neural Identity Matrix V24.35 (Modified) ---
