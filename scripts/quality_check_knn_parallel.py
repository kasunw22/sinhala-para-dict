"""
Evaluate the dataset using K-NN lookup method described in the paper
"""

import locale
locale.getpreferredencoding = lambda: "UTF-8"


import io
import os
import torch
import pickle


def save_var(var, f_name):
    # Open a file and use dump()
    with open(f'{f_name}', 'wb') as file:
        
        # A new file will be created
        pickle.dump(var, file)

    print(f"Saved {f_name}...")


def load_var(f_name):
    # Open the file in binary mode
    with open(f_name, 'rb') as file:
        
        # Call load method to deserialze
        var = pickle.load(file)
    
    print(f"Loaded {f_name}...")

    return var


# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {device}")

# En embedsdings setup
en_embedding_model_path = "/home/kasun/Documents/MSc/Sem3/Research/codes/Fasttext/fastText-master-kasun/alignment/data/cc.en.300.vec"
en_normalized_embd_path = '/home/kasun/Documents/MSc/Sem3/Research/codes/Fasttext/fastText-master-kasun/alignment/data/normalized_embd_en.pt'
en_words_path = '/home/kasun/Documents/MSc/Sem3/Research/codes/Fasttext/fastText-master-kasun/alignment/data/words_en.pt'

if os.path.exists(en_normalized_embd_path) and os.path.exists(en_words_path):
    print("[INFO] Loading EN distance tensors from existing files...")

    # cos_sim_mat_en = torch.load(en_cos_sim_mat_path)
    embeddings_en = torch.load(en_normalized_embd_path)
    words_en = torch.load(en_words_path)

else:
    if not os.path.exists(en_embedding_model_path):
        print("[INFO] Downloading EN embedding model...")

        os.system("wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz")
        os.system("gzip -d cc.en.300.vec.gz")
        os.system("cp cc.en.300.vec /home/kasun/Documents/MSc/Sem3/Research/codes/Fasttext/fastText-master-kasun/alignment/data/")

    print("[INFO] Processing embedding matrix for EN...")

    f = io.open(en_embedding_model_path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n_words_en, n_dim_en = map(int, f.readline().split())

    words_en = []
    embeddings_en = torch.zeros((n_words_en, n_dim_en), dtype=torch.float16).to(device)

    print(embeddings_en.shape, type(embeddings_en), embeddings_en.device)

    for i, line in enumerate(f):
        tokens = line.rstrip().split(' ')

        words_en.append(tokens[0])
        embeddings_en[i] = torch.tensor(list(map(float, tokens[1:])), dtype=torch.float16).to(device)

    f.close()

    print(embeddings_en.shape, type(embeddings_en), embeddings_en.device)

    with torch.no_grad():
        embeddings_en = torch.nn.functional.normalize(embeddings_en)

    torch.save(embeddings_en, en_normalized_embd_path)
    torch.save(words_en, en_words_path)

print("En-embd: ", embeddings_en.shape, type(embeddings_en), embeddings_en.device, len(words_en))

torch.cuda.empty_cache()

# Si embedsdings setup
si_embedding_model_path = "/home/kasun/Documents/MSc/Sem3/Research/codes/Fasttext/fastText-master-kasun/alignment/data/cc.si.300.vec"
si_normalized_embd_path = '/home/kasun/Documents/MSc/Sem3/Research/codes/Fasttext/fastText-master-kasun/alignment/data/normalized_embd_si.pt'
si_words_path = '/home/kasun/Documents/MSc/Sem3/Research/codes/Fasttext/fastText-master-kasun/alignment/data/words_si.pt'

if os.path.exists(si_normalized_embd_path) and os.path.exists(si_words_path):
    print("[INFO] Loading SI distance tensors from existing files...")

    embeddings_si = torch.load(si_normalized_embd_path)
    words_si = torch.load(si_words_path)

else:
    if not os.path.exists(si_embedding_model_path):
        print("[INFO] Downloading SI embedding model...")

        os.system("wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.si.300.vec.gz")
        os.system("gzip -d cc.si.300.vec.gz")
        os.system(f"cp cc.si.300.vec /home/kasun/Documents/MSc/Sem3/Research/codes/Fasttext/fastText-master-kasun/alignment/data/")

    print("[INFO] Processing embedding matrix for SI...")

    fs = io.open(si_embedding_model_path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n_words_si, n_dim_si = map(int, fs.readline().split())

    words_si = []
    embeddings_si = torch.zeros((n_words_si, n_dim_si), dtype=torch.float16).to(device)

    print(embeddings_si.shape, type(embeddings_si), embeddings_si.device)

    for i, line in enumerate(fs):
        tokens = line.rstrip().split(' ')

        words_si.append(tokens[0])
        embeddings_si[i] = torch.tensor(list(map(float, tokens[1:])), dtype=torch.float16).to(device)

    fs.close()

    print(embeddings_si.shape, type(embeddings_si), embeddings_si.device)

    with torch.no_grad():
        embeddings_si = torch.nn.functional.normalize(embeddings_si)
        
    torch.save(embeddings_si, si_normalized_embd_path)
    torch.save(words_si, si_words_path)


print("Si-embd: ", embeddings_si.shape, type(embeddings_si), embeddings_si.device, len(words_si))

# cache variables to save operations
en_cache_file = "/home/kasun/Documents/MSc/Sem3/Research/codes/Fasttext/fastText-master-kasun/alignment/data/en_kk_cache.pkl"
si_cache_file = "/home/kasun/Documents/MSc/Sem3/Research/codes/Fasttext/fastText-master-kasun/alignment/data/si_kk_cache.pkl"

if os.path.exists(en_cache_file):
    en_cache = load_var(en_cache_file)
else:
    en_cache = {}

if os.path.exists(si_cache_file):
    si_cache = load_var(si_cache_file)
else:
    si_cache = {}

en_cache_hits = 0
si_cache_hits = 0
cache_size = 100000

import threading


def clean_cache(cache):

    while len(cache) > cache_size:
        cache.pop(next(iter(cache)))

    torch.cuda.empty_cache()

def reorder_cache(si_cache, word):
    val = si_cache.pop(word)
    si_cache[word] = val
    torch.cuda.empty_cache()


@torch.no_grad()
def nearest_neighbors(word, src_lang_code="en", k=10):
    global en_cache_hits
    global si_cache_hits

    # Get cosinme distance vector for word
    if (src_lang_code == "en" and (word in words_en)) or (src_lang_code == "si" and (word in words_si)):

        if src_lang_code == "en":

            if word in en_cache:
                neighbors = en_cache[word]
                en_cache_hits += 1

                cl_th = threading.Thread(target=reorder_cache, args=(en_cache, word))
                cl_th.start()
            else:  
                word_idx = words_en.index(word)
                word_vec = embeddings_en[word_idx]

                cos_sim = torch.matmul(embeddings_en, word_vec)
                topk = torch.topk(cos_sim, k=k+1, dim=0)[1][1:]
                neighbors = [words_en[id_] for id_ in topk.cpu().numpy()]
                en_cache[word] = neighbors
                
                cl_th = threading.Thread(target=clean_cache, args=(en_cache,))
                cl_th.start()

        elif src_lang_code == "si":

            if word in si_cache:

                neighbors = si_cache[word]
                si_cache_hits += 1

                cl_th = threading.Thread(target=reorder_cache, args=(si_cache, word))
                cl_th.start()
            else:
                word_idx = words_si.index(word)
                word_vec = embeddings_si[word_idx]
                
                cos_sim = torch.matmul(embeddings_si, word_vec)
                topk = torch.topk(cos_sim, k=k+1, dim=0)[1][1:]
                neighbors = [words_si[id_] for id_ in topk.cpu().numpy()]
                si_cache[word] = neighbors

                cl_th = threading.Thread(target=clean_cache, args=(si_cache,))
                cl_th.start()

        else:
            return []

    else:
        return []

    return neighbors

en_cache
en_cache_hits

# Example usage
word = 'Hitler'
neighbors = nearest_neighbors(word, src_lang_code="en")
print(f'The 10 nearest neighbors of {word} are:')
for neighbor in neighbors:
    print(neighbor)

from string import punctuation
import xml.etree.ElementTree as ET
import spacy

# stop words
en = spacy.load('en_core_web_sm')
en_sw_list = en.Defaults.stop_words

# si_sw_file = "/content/drive/MyDrive/si-stop-words.txt"
si_sw_file = "/home/kasun/Documents/MSc/Sem3/Research/codes/sinhala/full-data/si-stop-words.txt"

with open(si_sw_file, encoding='utf-8') as f:
    si_sw_list = f.readlines()

si_sw_list = [w.strip() for w in si_sw_list]

def is_ASCII(word):
    is_ascii = [ord(x)<256 for x in word]
    
    if any(is_ascii):
        return True
    
    return False


def find_score(src_sen, tgt_sen, ref_dictionary, ref_keys, src_sw_list, src_lang_code, tgt_lang_code):
    score = 0
    count = 0
    score_wo_sw = 0
    count_wo_sw = 0
    no_entries = True

    if not (src_sen and tgt_sen):
        return score, score_wo_sw, no_entries

    src_words = src_sen.strip().translate(str.maketrans("", "", punctuation)).split()
    
    tgt_words = tgt_sen.strip().translate(str.maketrans("", "", punctuation)).split()
    # print(f"tgt_words: {tgt_words}")
    tgt_words_lower = [w.lower() for w in tgt_words if is_ASCII(w)]
    # print(f"tgt_words_lower: {tgt_words_lower}")
    
    tgt_words.extend(tgt_words_lower)
    # print(f"tgt_words: {tgt_words}")

    is_sw = False

    for src_w in src_words:

        is_sw = src_w in src_sw_list
        
        if (src_w in ref_keys) and (src_w not in tgt_words):
            no_entries = False

            tgt_w_list = ref_dictionary[src_w]

            if any(tgt_w in tgt_words for tgt_w in tgt_w_list):
                score += 1

                if not is_sw:
                    score_wo_sw += 1

            else:
              for tgt_w in tgt_w_list:
                nn_words = nearest_neighbors(tgt_w, tgt_lang_code)

                if any(nn_w in tgt_words for nn_w in nn_words):
                  score += 1

                  if not is_sw:
                      score_wo_sw += 1
                  
                  break
            
            count += 1

            if not is_sw:
                count_wo_sw += 1

        elif src_lang_code == "en":
            src_lower = src_w.lower()

            if (src_lower in ref_keys) and (src_lower not in tgt_words):
                no_entries = False

                tgt_w_list = ref_dictionary[src_lower]

                if any(tgt_w in tgt_words for tgt_w in tgt_w_list):
                    score += 1

                    if not is_sw:
                      score_wo_sw += 1

                else:
                  
                  for tgt_w in tgt_w_list:
                    nn_words = nearest_neighbors(tgt_w, tgt_lang_code)

                    if any(nn_w in tgt_words for nn_w in nn_words):
                      score += 1

                      if not is_sw:
                        score_wo_sw += 1

                      break
                
                count += 1

                if not is_sw:
                    count_wo_sw += 1
    
    score = 0 if (count == 0) else (score / count)
    score_wo_sw = 0 if (count_wo_sw == 0) else (score_wo_sw / count_wo_sw)

    return score, score_wo_sw, no_entries

# dictionary_file = "/content/drive/MyDrive/datasets-en-si-dict/V2/d3-sorted-filtered-Full-2.txt"
# dictionary_file = "/home/kasun/Documents/MSc/Sem3/Research/Multilingual-Embedding-Alignment/Papers/Facct-ACM-dataset-paper/datasets/V2/d3-sorted-filtered-Full-2.txt"
# dictionary_file = "/home/kasun/Documents/MSc/Sem3/Research/Multilingual-Embedding-Alignment/Papers/Facct-ACM-dataset-paper/datasets/V2/d2-sorted-filtered-Full-2.txt"
# dictionary_file = "/home/kasun/Documents/MSc/Sem3/Research/Multilingual-Embedding-Alignment/Papers/Facct-ACM-dataset-paper/datasets/V2/d1-sorted-filtered-Full-2.txt"
# dictionary_file = "/home/kasun/Documents/MSc/Sem3/Research/Multilingual-Embedding-Alignment/Papers/Facct-ACM-dataset-paper/datasets/V1/En-Si-dict-FastText.txt"
# dictionary_file = "/home/kasun/Documents/MSc/Sem3/Research/Multilingual-Embedding-Alignment/Papers/Facct-ACM-dataset-paper/datasets/V1/En-Si-dict-filtered.txt"
dictionary_file = "/home/kasun/Documents/MSc/Sem3/Research/Multilingual-Embedding-Alignment/Papers/Facct-ACM-dataset-paper/datasets/V1/En-Si-dict-large.txt"

en_si_dictionary = {}
si_en_dictionary = {}

print(f"Dataset: {dictionary_file}")

fd = open(dictionary_file, encoding='utf-8')

for entry in fd:
    en_w, si_w = entry.strip().split()

    if si_w not in si_en_dictionary:
        si_en_dictionary[si_w] = []
    
    si_en_dictionary[si_w].append(en_w)

    if en_w not in en_si_dictionary:
        en_si_dictionary[en_w] = []
        en_si_dictionary[en_w].append(si_w)

    else:
        en_si_dictionary[en_w].append(si_w)


en_si_keys = list(en_si_dictionary.keys())
si_en_keys = list(si_en_dictionary.keys())


dataset_name = "ted2020"
# dataset_name = "wikimedia"

if dataset_name == "wikimedia":
    dataset_file = "/home/kasun/Documents/MSc/Sem3/Research/Multilingual-Embedding-Alignment/Papers/Facct-ACM-dataset-paper/eval-datasets/WikiMedia/data"
    download_url = "https://object.pouta.csc.fi/OPUS-wikimedia/v20210402/tmx/en-si.tmx.gz"

elif dataset_name == "ted2020":
    dataset_file = "/home/kasun/Documents/MSc/Sem3/Research/Multilingual-Embedding-Alignment/Papers/Facct-ACM-dataset-paper/eval-datasets/TED2020-v1/data"
    download_url = "https://object.pouta.csc.fi/OPUS-TED2020/v1/tmx/en-si.tmx.gz"

if not os.path.exists(dataset_file):
    # download evaluation dataset
    print(f"[INFO] Downloading the {dataset_name} dataset...")

    os.system(f"wget {download_url}")
    os.system("gzip -d en-si.tmx.gz")
    os.system(f"cp en-si.tmx {dataset_file}")

tree = ET.parse(dataset_file)
root = tree.getroot()

print(root.tag)
print(root.attrib)


en_sen = None
si_sen = None

en_si_full_score = 0
en_si_full_count = 0
en_si_full_score_wo_sw = 0
en_si_full_count_wo_sw = 0

si_en_full_score = 0
si_en_full_count = 0
si_en_full_score_wo_sw = 0
si_en_full_count_wo_sw = 0

for i, tuv_tag in enumerate(root.iter('tuv')):
    sen = tuv_tag[0]
    
    if i%2 == 0:
        en_sen = sen.text
    else:
        si_sen = sen.text

        en_si_score, en_si_score_wo_sw, en_si_no_entries = find_score(en_sen, si_sen, en_si_dictionary, en_si_keys, en_sw_list, "en", "si")
        si_en_score, si_en_score_wo_sw, si_en_no_entries = find_score(si_sen, en_sen, si_en_dictionary, si_en_keys, si_sw_list, "si", "en")
        
        if not en_si_no_entries:
            en_si_full_score += en_si_score
            en_si_full_count += 1

            en_si_full_score_wo_sw += en_si_score_wo_sw
            en_si_full_count_wo_sw += 1

        if not si_en_no_entries:
            si_en_full_score += si_en_score
            si_en_full_count += 1

            si_en_full_score_wo_sw += si_en_score_wo_sw
            si_en_full_count_wo_sw += 1


    if (i+1)%1000 == 0:
        print("===" * 20)
        print(f"\nProcessed {(i+1) / 2} pairs. ...")

        print("With stop-words...")
        print("En-Si:")
        print(f"Cum score: {en_si_full_score}, Sentence pairs: {en_si_full_count}")
        print(f"Final average score: {en_si_full_score / en_si_full_count}")

        print("\nSi-En:")
        print(f"Cum score: {si_en_full_score}, Sentence pairs: {si_en_full_count}")
        print(f"Final average score: {si_en_full_score / si_en_full_count}")

        print("---" * 20)
        print("Without stop-words...")
        print("En-Si:")
        print(f"Cum score: {en_si_full_score_wo_sw}, Sentence pairs: {en_si_full_count_wo_sw}")
        print(f"Final average score: {en_si_full_score_wo_sw / en_si_full_count_wo_sw}")

        print("\nSi-En:")
        print(f"Cum score: {si_en_full_score_wo_sw}, Sentence pairs: {si_en_full_count_wo_sw}")
        print(f"Final average score: {si_en_full_score_wo_sw / si_en_full_count_wo_sw}")

print("#####" * 20)
print("With stop-words...")
print("En-Si:")
print(f"Cum score: {en_si_full_score}, Sentence pairs: {en_si_full_count}")
print(f"Final average score: {en_si_full_score / en_si_full_count}")

print("\nSi-En:")
print(f"Cum score: {si_en_full_score}, Sentence pairs: {si_en_full_count}")
print(f"Final average score: {si_en_full_score / si_en_full_count}")

print("---" * 20)
print("Without stop-words...")
print("En-Si:")
print(f"Cum score: {en_si_full_score_wo_sw}, Sentence pairs: {en_si_full_count_wo_sw}")
print(f"Final average score: {en_si_full_score_wo_sw / en_si_full_count_wo_sw}")

print("\nSi-En:")
print(f"Cum score: {si_en_full_score_wo_sw}, Sentence pairs: {si_en_full_count_wo_sw}")
print(f"Final average score: {si_en_full_score_wo_sw / si_en_full_count_wo_sw}")

print(f"\nen_cache_hits: {en_cache_hits}, si_cache_hits: {si_cache_hits}")

save_var(en_cache, en_cache_file)
save_var(si_cache, si_cache_file)
