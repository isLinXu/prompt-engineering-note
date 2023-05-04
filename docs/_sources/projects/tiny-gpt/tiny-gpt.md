# 使用pytorch从零开始实现一个简单的gpt

本文由prompt引导ChatGPT生成简易版gpt模型，根据比较关心的问题，使用了以下的prompt进行内容和代码的生成：

- prompt:->如何使用transformer模型在pytorch上进行大语言模型的设计实现，请分段给出完整的可行性实现
- prompt:->为了使用这个模型，应该如何使用制作自己的数据集
- prompt:->如何引入RLHF范式对以上模型进行迭代优化，给出完整实现代码和解释

注意：

- 这个测试是[prompt-engineering-note](https://github.com/isLinXu/prompt-engineering-note)项目的一部分；
- 这份生成的代码仅供参考与学习理解，目前仍在验证测试中，部分代码依旧需要人工干预微调，如果能够跑通，会另外新建一个GitHub项目进行开源。

![](https://img2023.cnblogs.com/blog/1571518/202305/1571518-20230504213614235-1155898481.png)

```
chatgpt prompt:->如何使用transformer模型在pytorch上进行大语言模型的设计实现，请分段给出完整的可行性实现
```


```
以下是PyTorch上使用transformer模型进行大语言模型设计实现的可行性实现步骤：

1.准备数据集

要训练大语言模型，首先需要准备一个足够大的文本语料作为数据集。可以使用公开的语料库，如维基百科、新闻文章等。在获取语料库后，需要将其进行预处理，将其转换为模型所需的格式。

2.数据预处理

对于每个文本样本，需要将其进行分词，并将每个词转换为对应的整数标识。可以使用已有的工具库，如NLTK、spaCy等完成分词操作。然后，可以使用PyTorch中的Tokenizer类将每个词转换为整数标识。还需要将文本划分为多个片段，每个片段的长度为固定的值，以便输入到模型中。

3.创建数据加载器

在PyTorch中，可以使用DataLoader类生成可以喂给模型的数据，该类可以对原本的数据进行批量化、随机化等处理。需要将预处理好的数据转换成PyTorch中的Tensor类型，并使用Dataset和DataLoader类分别封装数据集和数据加载器。

4.创建模型

接下来，需要创建一个Transformer模型。可以考虑使用已经在PyTorch中实现过的Transformer模型，如nn.TransformerEncoder。可以根据实际情况对模型进行适当的调整，调整模型层数、隐藏层大小、注意力头数等参数。

5.定义损失函数和优化器

在训练前，需要先定义损失函数和优化器。对于大语言模型来说，通常使用交叉熵损失函数作为模型的损失函数。可以使用PyTorch中的nn.CrossEntropyLoss类。优化器可以选择Adam等常用优化器。

6.训练模型

在定义好损失函数和优化器后，可以开始训练模型。可以使用PyTorch中的nn.Module类作为基类，自定义一个继承该类的模型。首先将模型设置为训练模式，然后对数据集进行遍历，在每个批次上计算损失并更新模型参数。

7.评估模型

在训练完成后，可以对模型进行评估。可以使用之前预处理好的数据集作为模型的输入，对模型进行推理并计算准确率、困惑度等模型评估指标。

8.应用模型

训练好的模型可以用于生成文本、机器翻译、语音识别等应用场景。可以使用PyTorch中的torch.save函数将模型保存到硬盘上，在需要应用模型时再加载。对于生成文本的任务，可以选择使用贪心搜索、束搜索等算法生成文本。
```

## 1.准备数据

首先需要准备训练模型所需要的数据，比如说可以选择WikiText-103这个数据集来训练大语言模型。可以使用PyTorch内置的torchtext库来加载数据集，具体实现可以参考下面的代码：

```python
from torchtext.datasets import WikiText103
from torchtext.data.utils import get_tokenizer

tokenizer = get_tokenizer('basic_english')
train_iter = WikiText103(split='train', tokenizer=tokenizer)
valid_iter = WikiText103(split='valid', tokenizer=tokenizer)
test_iter = WikiText103(split='test', tokenizer=tokenizer)
```

## 2.建立词表

接下来，我们需要把文本数据转换成数值表示，这时需要建立一个词表。可以使用PyTorch内置的Vocab类定义一个词表，并根据训练集构建词表。

```python
from torchtext.vocab import Vocab
import collections

counter = collections.Counter()
for data in train_iter:
    counter.update(data)
vocab = Vocab(counter)
```

## 3.定义模型

定义Transformer模型，可以使用PyTorch官方实现的torch.nn.Transformer类来构建。

```python

import math

import torch
import torch.nn as nn
from torch.nn import Transformer
from torchvision.models.video.mvit import PositionalEncoding

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, nhead, nhid, nlayers, dropout=0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.embedding_dim = embedding_dim
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(embedding_dim, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, nlayers)
        self.decoder = nn.Linear(embedding_dim, vocab_size)
        
        self.src_mask = None
        
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        
        src = self.pos_encoder(src * math.sqrt(self.embedding_dim))
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output.transpose(0,1)
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
```

## 4.训练模型

定义好模型后，就可以通过定义损失函数和优化器来训练模型。可以使用PyTorch内置的CrossEntropyLoss()作为损失函数，使用Adam优化器来更新模型参数。具体实现可以参考下面的代码：

```python
import torch.optim as optim
import math
import time

embed_size = 256
heads = 4
hidden_size = 256
num_layers = 4
dropout = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ntokens = len(vocab.stoi)
model = TransformerModel(ntokens, embed_size, heads, hidden_size, num_layers, dropout).to(device)

criterion = nn.CrossEntropyLoss()
lr = 5.0
optimizer = optim.Adam(model.parameters(), lr=lr)

def train(epoch):
    model.train()
    total_loss = 0.
    start_time = time.time()
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    for batch, data in enumerate(train_data):
        src = data.text.to(device)
        tgt = data.target.to(device)
        optimizer.zero_grad()
        if src.size(0) != bptt:
            src_mask = model.generate_square_subsequent_mask(src.size(0)).to(device)
        output = model(src)
        loss = criterion(output.view(-1, ntokens), tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch &gt; 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // bptt, lr,
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
            
def evaluate(eval_model, data_source):
    eval_model.eval()
    total_loss = 0.
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i, data in enumerate(data_source):
            src = data.text.to(device)
            tgt = data.target.to(device)
            if src.size(0) != bptt:
                src_mask = model.generate_square_subsequent_mask(src.size(0)).to(device)
            output = eval_model(src)
            output_flat = output.view(-1, ntokens)
            total_loss += len(src) * criterion(output_flat, tgt).item()
    return total_loss / (len(data_source) - 1)

best_val_loss = float("inf")
epochs = 20
bptt = 35

train_data = batchify(train_iter, bptt, device)
val_data = batchify(valid_iter, bptt, device)
test_data = batchify(test_iter, bptt, device)

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(epoch)
    val_loss = evaluate(model, val_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s |
valid loss {:5.2f} | valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                val_loss, math.exp(val_loss)))
    print('-' * 89)

    if val_loss &lt; best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'model.pth')
    test_loss = evaluate(model, test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
    
model.load_state_dict(torch.load('model.pth'))

test_loss = evaluate(model, test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
```

## 5.使用模型进行生成

训练好模型之后可以使用模型来生成文本。可以使用temperature对生成的结果进行控制，temperature越小，生成结果越保守，反之越激进。具体实现可以参考下面的代码：

```python
def generate_sentence(model, prompt, max_len=100, temperature=1.0):
    model.eval()
    with torch.no_grad():
        token = vocab.stoi[prompt]
        output_str = prompt
        inp = torch.LongTensor([[token]]).to(device)

        for i in range(max_len):
            output = model(inp)
            word_weights = output[-1].squeeze().div(temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            token = word_idx.item()

            if token == vocab.stoi['&lt;eos&gt;']:
                break

            output_str += vocab.itos[token]

            inp.fill_(token)

        return output_str
```

通过以上的步骤，就可以完成使用Transformer模型在PyTorch上进行大语言模型的设计实现。





```
chatgpt prompt:为了使用这个模型，应该如何使用制作自己的数据集
```

## 自定义数据集

为了使用这个模型，并训练自己的大型语言模型，你需要先准备自己的数据集。这里提供一个基本的数据集制作步骤：

1. 数据收集 首先，需要一些数据来训练大语言模型。可以使用公开数据集，或从文本文档、新闻、书籍、电子邮件等文本源中获取数据。它们应该是一个大的文本文件，其中每个句子/段/章节应该以换行符分隔，并带有标点符号。

2.  清理并训练文本 在构建一个训练模型之前，通常需要清理和格式化收集到的数据。通常，文本已经被清理好，但仍需要进一步的处理，如文件分割，并将其准备成经过格式化的数据集。 对于大型数据集，最好将文本分成小块使加载速度更快。使用Python的split()或其他分割数据的方法进行分割。 

3. 创建训练集、验证集和测试集 将数据集分成三个部分：训练集、验证集和测试集。通常使用80%的数据作为训练集、10%的数据作为验证集和10%的数据作为测试集。这些数据集的目的是评估模型的性能，以及确定训练的最佳超参数设置。 

4. 建立词表 如上所述，需要将文本数据转换为数字数据表示。需要将文本数据中可能包含的所有单词映射到唯一的整数。我们称这些整数为词汇表中的单词标识符。这一步，我们需要建立一个词表，并根据训练集构建词表。可以使用Python的collections.Counter()来统计单词出现的次数，然后使用PyTorch中的Vocab()来生成词表。 

5. 将文本转换为数字序列 在训练和测试模型时，需要使用数字数据来代替文本数据进行操作。这通常可以通过将词汇表中的单词标识符与句子中的单词进行匹配，并替换每个单词为所属的唯一整数来实现。 

6. 构建批次 最后一步是创建批次，也就是将序列打包成固定大小的批次进行训练。在批次中，每个句子的长度必须相同。可以使用PyTorch的DataLoader类来完成这个任务。 参考上述步骤，你可以以类似WikiText-103的格式创建自己的训练集，并使用Transformer模型在PyTorch上训练您自己的大型语言模型。

   

```
chatgpt prompt:->如何引入RLHF范式对以上模型进行迭代优化，给出完整实现代码和解释
```

## 如何使用RLHF进行强化学习

RLHF（Reinforcement Learning with Hierarchical Feedback）是一种强化学习范式，它在原有强化学习框架基础上增加了一层更高的控制层，将原来的状态转移和奖励函数分别转化为学习过程和反馈过程。接下来，我们将介绍如何将RLHF框架应用到以上的Transformer模型上进行迭代优化。

1.定义参数

首先，定义模型所需的参数和超参数，包括词向量的维度、Transformer层数、头数、隐藏层的维度等。我们需要引入RLHF范式中的超参数，包括学习率、温度、奖励系数等。具体实现可以参考以下代码:

```
import torch.nn.functional as F

emb_size = 256
num_layers = 4
num_heads = 8
hidden_size = 512

lr = 0.005
temp = 1.0
reward_coef = 0.1
```

2.定义环境

接下来，我们需要将数据集转换为环境，其中每个状态是模型的输入，每个动作是生成文本序列中下一个单词。我们还需要定义环境中的奖励函数，即根据生成文本的质量给出奖励。具体实现可以参考以下代码：

```
import numpy as np

class LanguageModelEnv():
    def __init__(self, model, vocab, seq_len):
        self.model = model
        self.vocab = vocab
        self.seq_len = seq_len
        self.reset()

    def reset(self):
        self.generated = ""
        self.observation = ""
        self.state = torch.zeros(1, self.seq_len, dtype=torch.long)
        self.i = 0
        self.done = False
        return self.observation

    def step(self, action):
        if self.done:
            return self.observation, 0, self.done, {}
        token = self.vocab.stoi[action]
        output, _ = self.model(self.state)
        prob = F.softmax(output[-1, :], dim=0).cpu().detach().numpy()
        next_token = np.random.choice(len(self.vocab), p=prob)
        self.generated += action
        self.state[0, :-1] = self.state[0, 1:]
        self.state[0, -1] = torch.tensor([next_token])
        self.i += 1
        if self.i == self.seq_len:
            self.done = True
            reward = self.get_reward(self.generated)
        else:
            reward = 0
        self.observation = self.generated[-1]
        return self.observation, reward, self.done, {}

    def get_reward(self, generated_text):
        with torch.no_grad():
            input_ids = torch.tensor([self.vocab.stoi['&lt;bos&gt;']] + [self.vocab.stoi[word] for word in generated_text.split()])
            input_ids = input_ids.unsqueeze(0).to(self.model.device)
            output = self.model(input_ids)
            log_prob = F.log_softmax(output[0, :-1, :], dim=-1)
            lg_p_nxt_word = log_prob[range(len(generated_text)), input_ids[0, 1:]].sum().item()
            reward = (lg_p_nxt_word - np.log(1/len(self.vocab))) * reward_coef
        return reward
```



3. 定义RLHF框架的训练过程

下一步，我们需要定义整个RLHF框架的训练过程。训练过程中，我们要先定义一个基本的强化学习算法，即使用蒙特卡洛策略梯度更新。同时，我们要建立一个控制层，用来监测上层的状况，并给出最优的模型参数，在下一轮迭代中更新模型参数。具体实现可以参考以下代码：

```
import copy

def policy_gradient(env, policy, optimizer):
    rewards = []
    log_probs = []
    obs = env.reset()
    done = False
  
    while not done:
        action, log_prob = policy(obs, temp)
        obs, reward, done, _ = env.step(action)
        log_probs.append(log_prob)
        rewards.append(reward)

    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + R
        returns.append(R)

    returns = np.array(returns[::-1])

    log_probs = torch.stack(log_probs)
    policy_loss = -(log_probs * returns).sum()

    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

    return policy_loss.item()

def control_loop(env, num_iter, num_samples):
    policy = lambda x, temp: policy_fn(model, x, vocab, temp)
    best_params = copy.deepcopy(model.state_dict())
    best_score = 0.0
  
    for i in range(num_iter):
        model.load_state_dict(best_params)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for _ in range(num_samples):
            loss = policy_gradient(env, policy, optimizer)

        score = evaluate(env, model, test_data)
        if score &gt; best_score:
            best_score = score
            best_params = copy.deepcopy(model.state_dict())

        print(f'Iteration {i+1}: score={score:.2f}')
  
    model.load_state_dict(best_params)
```

这段代码实现了策略梯度（policy gradient）算法的控制循环（control loop）。下面对代码进行一些解释：

1. policy_gradient函数接受一个环境对象env，一个策略函数policy以及一个优化器optimizer作为输入，并返回一个policy_loss值（即策略梯度算法的损失函数）。

2. control_loop函数接受一个环境对象env、一个整数num_iter（表示控制循环迭代的次数）和一个整数num_samples（表示每次迭代中样本的数量）。

3. 在控制循环函数中，使用lambda表达式定义一个策略函数policy，该函数接受一个状态变量x和一个温度变量temp，并返回该状态下执行每个可能动作的概率。

4. 首先使用深拷贝函数copy.deepcopy将模型的参数复制到best_params中，并将best_score初始化为0.0。

5. 在每次控制循环中进行num_samples个样本的策略梯度下降更新，并将返回的policy_loss值累加。

6. 在每次迭代结束后，通过evaluate函数评估当前模型在测试数据上的表现，并将当前模型的状态与最佳模型的状态进行比较。如果当前模型比最佳模型表现更好，就更新最佳模型的参数和最佳分数。

7. 打印出每次控制循环的迭代次数和当前分数。

8. 最后加载最佳参数，并返回训练好的模型。

注：代码中所示的“&gt;”实际上应该是“&gt;”，由于网站自动替换了符号。