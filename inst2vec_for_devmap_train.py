import os
import math
import torch
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from src.utils.system import read_ir_from_file
from sklearn.model_selection import StratifiedKFold
from src.observation.inst2vec import Inst2vecEncoder

data_folder = 'data/opencl_device_mapping'
platform = 'all'
num_epochs = 50
batch_size = 64
dense_layer_size = 32
print_summary = False
out_folder = 'output/inst2vec_for_devmap'
num_classes = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
platform2str = {
    "amd": "AMD Tahiti 7970",
    "nvidia": "NVIDIA GTX 970"
}

if not os.path.exists(out_folder):
    os.makedirs(out_folder)
assert platform in ['all', 'amd', 'nvidia'], \
    'Choose device among: all, amd, nvidia'


class DevMapDataset(Dataset):
    def __init__(self, sequences, aux_in, y, embeddings):
        super().__init__()
        self.sequences = sequences
        self.aux_in = aux_in
        self.y = y
        self.embeddings = embeddings
        self.embedding_input = self.embeddings[self.sequences]
        
        
    def __getitem__(self, index):
        seqs = self.embedding_input[index]
        aux = self.aux_in[index]
        label = self.y[index]
        return seqs, aux, label

    def __len__(self):
        return len(self.y)


class DevMapLSTM(nn.Module):
    def __init__(self, embedding_dim, dense_layer_size):
        super(DevMapLSTM, self).__init__()
        self.lstm_1 = nn.LSTM(embedding_dim, embedding_dim, batch_first=True)
        self.lstm_2 = nn.LSTM(embedding_dim, embedding_dim, batch_first=True)
        self.language_model_out = nn.Linear(embedding_dim, 2)
        self.batch_norm = nn.BatchNorm1d(embedding_dim + 2)
        self.dense_1 = nn.Linear(embedding_dim + 2, dense_layer_size)
        self.output = nn.Linear(dense_layer_size, 2)
        
    def forward(self, x, aux_input):
        out, _ = self.lstm_1(x)
        out, _ = self.lstm_2(out)
        lang_output = torch.sigmoid(self.language_model_out(out[:, -1, :]))
        x_combined = torch.cat((aux_input, out[:, -1, :]), dim=1)
        x_combined = self.batch_norm(x_combined)
        x_combined = torch.relu(self.dense_1(x_combined))
        final_output = torch.sigmoid(self.output(x_combined))
        return final_output, lang_output
    

def load_data(data_path, platform):
    # Load runtime data
    df = pd.read_csv(data_path + "/cgo17-{}.csv".format(platform), index_col=0)
    print('--- Read data from', data_path)

    df["bench_data"] = (
        df.loc[df["dataset"] != "default", "benchmark"]
        + str("_")
        + df.loc[df["dataset"] != "default", "dataset"]
    )
    df.loc[df["dataset"] == "default", "bench_data"] = df.loc[
        df["dataset"] == "default", "benchmark"
    ]
    df["bench_data_path"] = data_path + '/kernels_ir/' + df["bench_data"] + str(".ll")

    # inst2vec embedding
    input_files = df["bench_data_path"].values   # list of benchmark file path
    num_files = len(input_files)
    num_unks = 0
    seq_lengths = list()
    encoder = Inst2vecEncoder()  # inst2vec 编码器
    unk_idx = encoder.unknown_vocab_element    
    print('--- Preparing to read', num_files, 'input files from folder', data_path + '/kernels_ir/')
    seqs = list()
    for i in tqdm(range(num_files), desc='Encoding files'):
        file = input_files[i]
        if os.path.exists(file):
            ir = encoder.preprocess(file)
            encode_ir = encoder.encode(ir)  # inst2vec编码
            seq_lengths.append(len(encode_ir))
            num_unks += encode_ir.count(str(unk_idx))
            seqs.append([int(s) for s in encode_ir])
        else:
            raise FileNotFoundError('Input file not found: ' + file)

    maxlen = max(seq_lengths)
    print('Number of benchmark  : {:>5}'.format(num_files))
    print('Shortest sequence    : {:>5}'.format(min(seq_lengths)))
    print('Longest sequence     : {:>5}'.format(maxlen))
    print('Mean sequence length : {:>5} (rounded down)'.format(math.floor(np.mean(seq_lengths))))
    print('Number of \'UNK\'      : {:>5}'.format(num_unks))
    print('Percentage of \'UNK\'  : {:>8.4} (% among all stmts)'.format((num_unks*100)/sum(seq_lengths)))
    print('\'UNK\' index          : {:>5}'.format(unk_idx))

    # Padding logic
    padded_sequences = []
    for seq in seqs:
        if len(seq) < maxlen:
            # Pad sequence if it is shorter than maxlen
            seq = seq + [unk_idx] * (maxlen - len(seq))
        padded_sequences.append(seq)

    # Convert to np.array
    encoded = np.array(padded_sequences)

    # aux data
    aux_in = np.array([
        df["transfer"].values,
        df["wgsize"].values,
    ]).T
    
    # 标签
    label = np.array([1 if x == "GPU" else 0 for x in df["oracle"].values])

    return encoded, aux_in, label, encoder.embeddings, df


def escape_suite_name(g: str) -> str:
    c = g.split('-')
    if c[0] == "amd" or c[0] == "nvidia":
        return c[0].upper() + " SDK"
    if c[0] == "npb" or c[0] == "shoc":
        return c[0].upper()
    elif c[0] == "parboil" or c[0] == "polybench" or c[0] == "rodinia":
        return c[0].capitalize()
    else:
        raise LookupError

def escape_benchmark_name(g: str) -> str:
    c = g.split('-')
    return escape_suite_name(c[0]).split()[0] + "." + c[-2]

def train_model(model, loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        correct = 0
        for batch in loader:
            sequences, aux_input, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs, lang_outputs = model(sequences, aux_input)
            loss = criterion(outputs, labels) + 0.2 * criterion(lang_outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
                    
        accuracy = correct / len(loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")


def evaluate(platform, data_folder, out_folder, dense_layer_size, num_epochs, batch_size):
    if platform == 'all':
        platform_list = ["amd", "nvidia"]
    else:
        platform_list = [platform]

    data = []
    for i, platform in enumerate(platform_list):
        # 读取数据集
        sequences, aux_in, y, embeddings, df = load_data(data_folder, platform)
        aux_in_tensor = torch.tensor(aux_in, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.int64)
        y_onehot_tensor = F.one_hot(y_tensor, num_classes=num_classes).to(torch.float32)

        # 使用 F.normalize 进行 L2 归一化
        embeddings = torch.tensor(embeddings, dtype=torch.float32)
        embedding_matrix_normalized = F.normalize(embeddings, p=2, dim=1)
        

        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=204)
        for j, (train_index, test_index) in enumerate(kf.split(sequences, y)):
            print('--- Cross validation step [', j, '/ 10 ]')

            model_basename = 'lstm'
            model_path = os.path.join(out_folder, f"models/{model_basename}-{platform}-{j}.pth")
            predictions_path = os.path.join(out_folder, f"predictions/{model_basename}-{platform}-{j}.result")
            log_dir = os.path.join(out_folder, "logs")

            if os.path.exists(predictions_path):
                print("\tFound predictions in", predictions_path, ", skipping...")
                with open(predictions_path, 'rb') as infile:
                    p = pickle.load(infile)
            else:

                if not os.path.exists(model_path):
                    
                    # 创建模型
                    model = DevMapLSTM(embedding_dim=embedding_matrix_normalized.shape[1], dense_layer_size=dense_layer_size).to(device)
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(model.parameters(), lr=0.001)

                    train_data = DevMapDataset(sequences[train_index], aux_in_tensor[train_index], y_tensor[train_index], embedding_matrix_normalized)
                    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

                    print('--- Training model... ')
                    train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs)
                    
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    torch.save(model.state_dict(), model_path)
                    print('--- Saved model to', model_path)
                    
                else:

                    # 读取模型权重文件
                    model = DevMapLSTM(embedding_dim=embedding_matrix_normalized.shape[1], dense_layer_size=dense_layer_size).to(device)
                    model.load_state_dict(torch.load(model_path))
                    print("Found trained model in", model_path, ", skipping...")
                    
                # 模型预测
                test_data = DevMapDataset(sequences[test_index], aux_in_tensor[test_index], y_tensor[test_index], embedding_matrix_normalized)
                test_loader = DataLoader(test_data, batch_size=batch_size)
                model.eval()
                p = []
                with torch.no_grad():
                    for batch in test_loader:
                        batch = [b.to(device) for b in batch]
                        output = model(batch[0], batch[1])
                        preds = torch.argmax(output[0], dim=-1).view(-1)
                        p.extend(preds.cpu().numpy())

                os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
                with open(predictions_path, 'wb') as outfile:
                    pickle.dump(p, outfile)
                print('\tsaved predictions to', predictions_path)


            benchmarks = df['benchmark'].values[test_index]
            o = y[test_index]
            correct = (np.array(p) == o)

            zero_r_dev = "runtime_cpu" if platform == "amd" else "runtime_gpu"
            zer_r_runtimes = df[zero_r_dev].values[test_index]
            runtimes = df[['runtime_cpu', 'runtime_gpu']].values[test_index]
            p_runtimes = [r[p_] for p_, r in zip(np.array(p, dtype=int), runtimes)]
            p_speedup = zer_r_runtimes / p_runtimes

            assert len(benchmarks) == len(o) == len(correct) == len(p) == len(p_speedup)

            for benchmark_, o_, p_, correct_, p_speedup_ in zip(benchmarks, o, p, correct, p_speedup):
                data.append({
                    "Model": model_basename,
                    "Platform": platform2str[platform],
                    'Benchmark': escape_benchmark_name(benchmark_),
                    'Benchmark Suite': escape_suite_name(benchmark_),
                    "Oracle Mapping": int(o_),
                    "Predicted Mapping": int(p_),
                    "Correct?": bool(correct_),
                    "Speedup": float(p_speedup_),
                })
            
    return pd.DataFrame(
        data, index=range(1, len(data) + 1), columns=[
            "Model",
            "Platform",
            "Benchmark",
            "Benchmark Suite",
            "Oracle Mapping",
            "Predicted Mapping",
            "Correct?",
            "Speedup"
        ])


if __name__ == '__main__':
    
    static_pred_vals = [58.823529, 56.911765]
    static_pred_mean = 57.867647
    static_sp_vals = [1.0, 1.0]
    static_sp_mean = 1.0
    grewe_pred_vals = [73.382353, 72.941176]
    grewe_pred_mean = 73.161765
    grewe_sp_vals = [2.905822, 1.264801]
    grewe_sp_mean = 2.085312
    deeptune_pred_vals = [83.676471, 80.294118]
    deeptune_pred_mean = 81.985294
    deeptune_sp_vals = [3.335612, 1.412222]
    deeptune_sp_mean = 2.373917

    print("Evaluating DeepTuneInst2Vec ...")
    result = evaluate('all', data_folder, out_folder, dense_layer_size, num_epochs, batch_size)

    # Print results
    print('\n--- Prediction results')
    print(result.groupby(['Platform', 'Benchmark Suite'])[['Correct?', 'Speedup']].mean())
    print('\n--- Prediction results (summarized)')
    print(result.groupby(['Platform'])[['Correct?', 'Speedup']].mean())

    # Model comparison: prediction accuracy
    print('\n--- Model comparison: prediction accuracy')
    d = list()
    d.append(np.append(static_pred_vals, static_pred_mean))
    d.append(np.append(grewe_pred_vals, grewe_pred_mean))
    d.append(np.append(deeptune_pred_vals, deeptune_pred_mean))
    d.append(np.append(result.groupby(['Platform'])['Correct?'].mean().values * 100,
                        result['Correct?'].mean() * 100))
    d = np.array(d).T.reshape(3, 4)
    print('\n', pd.DataFrame(d, columns=['Static mapping', 'Grewe et al.', 'DeepTune', 'DeepTuneInst2Vec'],
                                index=['AMD Tahiti 7970', 'NVIDIA GTX 970', 'Average']))

    # Model comparison: speedups
    print('\n--- Model comparison: speedups')
    d = list()
    d.append(np.append(static_sp_vals, static_sp_mean))
    d.append(np.append(grewe_sp_vals, grewe_sp_mean))
    d.append(np.append(deeptune_sp_vals, deeptune_sp_mean))
    d.append(np.append(result.groupby(['Platform'])['Speedup'].mean().values,
                        result['Speedup'].mean()))
    d = np.array(d).T.reshape(3, 4)
    print('\n', pd.DataFrame(d, columns=['Static mapping', 'Grewe et al.', 'DeepTune', 'DeepTuneInst2Vec'],
                                index=['AMD Tahiti 7970', 'NVIDIA GTX 970', 'Average']))






