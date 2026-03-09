import pandas as pd
import logging
import torch
import pickle
import questionary
from torch.utils.data import TensorDataset, DataLoader
from safetensors.torch import save_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from src.tokenizer import Tokenizer
from model import Model
from src.create_data import create_train_csv
from src.config import init_config, get_config

init_config()
config = get_config()

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

dataset_path = 'dataset/train.csv'

df = pd.read_csv(dataset_path)
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

logging.debug(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")


train_sentences = train_df['sentence'].tolist()
train_labels = train_df['label'].to_numpy().copy()

val_sentences = val_df['sentence'].tolist()
val_labels = val_df['label'].to_numpy().copy()

test_sentences = test_df['sentence'].tolist()
test_labels = test_df['label'].to_numpy().copy()

t = Tokenizer(config=get_config())

#語彙を構築
t.build_vocab(train_sentences)

#エンコード
train_X = [t.encode(s) for s in train_sentences]
val_X = [t.encode(s) for s in val_sentences]
test_X = [t.encode(s) for s in test_sentences]

#テンソル化
train_X = torch.LongTensor(train_X)
train_y = torch.FloatTensor(train_labels)

val_X = torch.LongTensor(val_X)
val_y = torch.FloatTensor(val_labels)

test_X = torch.LongTensor(test_X)
test_y = torch.FloatTensor(test_labels)

#データローダーの作成
train_dataset = TensorDataset(train_X, train_y)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


val_dataset = TensorDataset(val_X, val_y)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

vocab_size = len(t.word2idx) + 1
model = Model(
    vocab_size, 
    embedding_dim=config.get('embedding_dim', 128),
    num_filters=config.get('cnn_out_channels', 64),
    lstm_hidden=config.get('lstm_hidden_size', 128)
)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#テスト
test_sentence = "うせやろ？"
encoded = t.encode(test_sentence)
decoded = t.decode(encoded)

assert test_sentence[:t.max_len] == decoded, "Not match after encoding and decoding"


#訓練
def train():
    create_train_csv()
    
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            output = model(batch_X)
            output = output.squeeze()
            
            loss = criterion(output, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        logging.debug(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
        
        model.eval()
        with torch.no_grad():
            val_output = model(val_X)
            val_loss = criterion(val_output.squeeze(), val_y)
            logging.debug(f"Val Loss: {val_loss.item():.4f}")
            
    #torch.save(model.state_dict(), 'data/model.pth')
    try:
        save_file(model.state_dict(), "data/model.safetensors")
        logging.debug("Model saved to data/model.safetensors")
    except Exception as e:
        logging.error(f"Failed to save model: {e}")

    try:
        t.save("data/tokenizer.json")
        logging.debug("Tokenizer saved to data/tokenizer.json")
    except Exception as e:
        logging.error(f"Failed to save tokenizer: {e}")
        
incorrect_cnt = 10       

while True:
    try:
        if incorrect_cnt >= 10:
            train()
            incorrect_cnt = 0
        
        
        logging.info(f"Test iteration {incorrect_cnt}/10")
        
        user_text = questionary.text("Enter a sentence to test (or 'exit' to quit):").ask()
        if user_text.lower() == 'exit':
            train()    
            break
        elif user_text == '' or user_text == None:
            logging.info("Please enter a valid sentence.")
            continue
        
        encoded = t.encode(user_text)
        input_tensor = torch.LongTensor([encoded])
        with torch.no_grad():
            prob = model(input_tensor).item()
            
        if prob > 0.5:
            result = "淫夢語録"
        else:
            result = "通常文章"
        
        logging.info(f"Classified as: {result}")
        
        is_correct = questionary.select("Is the classification correct?", choices=["Yes", "No"]).ask()
        
        if is_correct == "No":
            #noだから逆のラベルを保存
            incorrect_cnt += 1
            
            correct_label = "通常文章" if result == "淫夢語録" else "淫夢語録"
            
            if correct_label == "淫夢語録":
                with open('dataset/raw/positive.txt', 'a', encoding='utf-8') as f:
                    f.write(user_text + '\n')
            else:
                with open('dataset/raw/negative.txt', 'a', encoding='utf-8') as f:
                    f.write(user_text + '\n')
            
            
    except KeyboardInterrupt:
        logging.info("Exiting...")
        train()
        break