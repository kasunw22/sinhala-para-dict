"""
Evaluate the dataset using simple lookup method described in the paper
"""

from string import punctuation
import xml.etree.ElementTree as ET

import spacy
en = spacy.load('en_core_web_sm')

en_sw_list = en.Defaults.stop_words

si_sw_file = "si-stop-words.txt"

with open(si_sw_file) as f:
    si_sw_list = f.readlines()

si_sw_list = [w.strip() for w in si_sw_list]

dictionary_file = "/home/kasun/Documents/MSc/Sem3/Research/Multilingual-Embedding-Alignment/Papers/Facct-ACM-dataset-paper/datasets/V1/En-Si-dict-FastText.txt"
dataset_file = "/home/kasun/Documents/MSc/Sem3/Research/Multilingual-Embedding-Alignment/Papers/Facct-ACM-dataset-paper/eval-datasets/TED2020-v1/data"

en_si_dictionary = {}
si_en_dictionary = {}

print(f"Dataset: {dictionary_file}")

fd = open(dictionary_file)

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


def is_ASCII(word):
    is_ascii = [ord(x)<256 for x in word]
    
    if any(is_ascii):
        return True
    
    return False


def find_score(src_sen, tgt_sen, ref_dictionary, ref_keys, src_sw_list, src_lang_code):
    score = 0
    count = 0
    score_wo_sw = 0
    count_wo_sw = 0
    no_entries = True

    if not (src_sen and tgt_sen):
        return score, no_entries

    src_words = src_sen.strip().translate(str.maketrans("", "", punctuation)).split()
    
    tgt_words = tgt_sen.strip().translate(str.maketrans("", "", punctuation)).split()
    tgt_words_lower = [w.lower() for w in tgt_words if is_ASCII(w)]
    
    tgt_words.extend(tgt_words_lower)

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
                
                count += 1

                if not is_sw:
                    count_wo_sw += 1
    
    score = 0 if (count == 0) else (score / count)
    score_wo_sw = 0 if (count_wo_sw == 0) else (score_wo_sw / count_wo_sw)

    return score, score_wo_sw, no_entries


tree = ET.parse(dataset_file)
root = tree.getroot()

print(root.tag)
print(root.attrib)

omit_sw = True
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

        en_si_score, en_si_score_wo_sw, en_si_no_entries = find_score(en_sen, si_sen, en_si_dictionary, en_si_keys, en_sw_list, "en", omit_sw)
        si_en_score, si_en_score_wo_sw, si_en_no_entries = find_score(si_sen, en_sen, si_en_dictionary, si_en_keys, si_sw_list, "si", omit_sw)

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

    if i%1000 == 0:

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
