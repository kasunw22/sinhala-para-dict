# sinhala-para-dict
English-Sinhala parallel word dictionary and resources

- scripts/si-stop-words.txt is taken from [this](https://github.com/nlpcuom/Sinhala-Stopword-list/blob/master/stop%20words.txt) repository.
- word-frequencies/all_counts.txt contains the word-frequency analysis we have done. We used [STM](https://github.com/scnakandala/sinhala_text_mining/tree/master) and [SinMin](https://rtuthaya.staff.uom.lk/resources/dataset/17) corpora
### BibTex Citation
If you are willing to use this work, please be kind enough to cite the following [paper](https://arxiv.org/abs/2308.02234).

```
@INPROCEEDINGS{Wick2308:Sinhala,
    AUTHOR="Kasun Wickramasinghe and Nisansa {de Silva}",
    TITLE="{Sinhala-English} Parallel Word Dictionary Dataset",
    BOOKTITLE="2023 IEEE 17th International Conference on Industrial and Information
    Systems (ICIIS) (ICIIS'2023)",
    ADDRESS="Peradeniya, Sri Lanka",
    PAGES=6,
    DAYS=24,
    MONTH=aug,
    YEAR=2023,
    KEYWORDS="Parallel Corpus; Alignment; English-Sinhala Dictionary; Word Embedding
    Alignment; Lexicon Induction",
    ABSTRACT="Parallel datasets are vital for performing and evaluating any kind of
    multilingual task. However, in the cases where one of the considered
    language pairs is a low-resource language, the existing top-down parallel
    data such as corpora are lacking in both tally and quality due to the
    dearth of human annotation. Therefore, for low-resource languages, it is
    more feasible to move in the bottom-up direction where finer granular pairs
    such as dictionary datasets are developed first. They may then be used for
    mid-level tasks such as supervised multilingual word embedding alignment.
    These in turn can later guide higher-level tasks in the order of aligning
    sentence or paragraph text corpora used for Machine Translation (MT). Even
    though more approachable than generating and aligning a massive corpus for
    a low-resource language, for the same reason of apathy from larger research
    entities, even these finer granular data sets are lacking for some
    low-resource languages. We have observed that there is no free and open
    dictionary data set for the low-resource language, Sinhala. Thus, in this
    work, we introduce three parallel English-Sinhala word dictionaries
    (En-Si-dict-large, En-Si-dict-filtered, En-Si-dict-FastText) which help in
    multilingual Natural Language Processing (NLP) tasks related to English and
    Sinhala languages. In this paper, we explain the dataset creation pipeline
    as well as the experimental results of the tests we have carried out to
    verify the quality of the data sets. The data sets and the related scripts
    are available at https://github.com/kasunw22/sinhala-para-dict/tree/main."
}
```
