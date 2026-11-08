"""
Stage 1: Building LLM
    1. Data preparation
    2. Attention Mechanism
    3. LLM architecture
        4. Pretraining

Stage 2: Foundation Model
    5. Training loop
    6. Model Evaluation
    7. Load Pretrained weights
        8. Fine-tuning
"""

"""

Data preparation

"""
#Get file
"""
import os
import urllib.request

if not os.path.exists("the-verdict.txt"):
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)
"""

#Open and print some to ensure its working fine
"""
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
print("Total number of character:", len(raw_text))
print(raw_text[:99])
"""

#Data tockenisation
#Split the text ===> word2Vec
"""
import re
text = "hello world. This is example text!"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
print(result)

"""

#Data tockenisation
#Split the text in entire file and strip to remove white spaces
import re
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
#print(preprocessed[:30])
#print(len(preprocessed))

#Create vocabulary by assigning each unique word an index
#store them in a set
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

#print(vocab_size)

vocab = {token:integer for integer,token in enumerate(all_words)}
#Below are the first 50 entries in this vocabulary:
for i, item in enumerate(vocab.items()):
    #print(item)
    if i >= 50:
        break


#putting together

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    #Process input text into tocken Id
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
                                
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    #convert token IDs back to text 
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
    
#We can use the tokenizer to encode (that is, tokenize) texts into integers
tokenizer = SimpleTokenizerV1(vocab)

text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
#print(ids)
#print(tokenizer.decode(ids))


#Hello was no used in original training data so it will result into an error
text = "hello, do you know me?"
#print(tokenizer.encode(text))

#=======================================================================================================================================================================

#Expand vocabulary to add new words and data
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])

vocab = {token:integer for integer,token in enumerate(all_tokens)}
#print(len(vocab.items()))

#print thelast 5 so we ca confirm new vocabulary was added
# for i, item in enumerate(list(vocab.items())[-5:]):
#     print(item)

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int 
            else "<|unk|>" for item in preprocessed #if input text is unknown
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        #join the tockens while adding white spaces
        text = " ".join([self.int_to_str[i] for i in ids])
        # Remove spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

#Let's try to tokenize text with the modified tokenizer:

tokenizer = SimpleTokenizerV2(vocab)
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
#print(text) #Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace.

#print(tokenizer.encode(text)) #[1131, 5, 355, 1126, 628, 975, 10, 1130, 55, 988, 956, 984, 722, 988, 1131, 7]
#print(tokenizer.decode(tokenizer.encode(text))) #<|unk|>, do you like tea? <|endoftext|> In the sunlit terraces of the <|unk|>.



#===============================================================BPE========================================================================

import importlib.metadata
import tiktoken

# print("tiktoken version:", importlib.metadata.version("tiktoken"))

bpe_tockenizer = tiktoken.get_encoding("gpt2")
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)

integers = bpe_tockenizer.encode(text, allowed_special={"<|endoftext|>"})

#print(integers)

#print(bpe_tockenizer.decode(integers))

#==========================================================Data sampling with sliding window==============================================================================================
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
enc_text = bpe_tockenizer.encode(raw_text)
    
print("Total number of character:", len(enc_text))

#Grab the the rest of the story after the first 50 tockens
enc_sample = enc_text[50:]

context_size = 4

x = enc_sample[:context_size] #input tocken
y = enc_sample[1:context_size+1] #Target tockens

#print(f"x: {x}")
#print(f"y:      {y}")

#here we are mimicing prediction to facilitate training so later the model can understand which tokens could be next in sequence

#taking a sample and sliding the window to reveal the next token
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]

    print(context, "---->", desired)

#now using decoded tokens
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]

    print(bpe_tockenizer.decode(context), "---->", bpe_tockenizer.decode([desired]))

#Create dataset and dataloader that extract chunks from the input text dataset
import torch
from torch.utils.data import Dataset, DataLoader

print("PyTorch version:", torch.__version__)

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > max_length, "Number of tokenized inputs must at least be equal to max_length+1"

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader