

<p align="center">
<img src="notebooks+utils+data\TextSummarizationImg.jpg" height=200 />
</p>

# <b>Abstract Text Summarization Using Transformers</b>
Hugging Face Transformers to build abstract text summarization NLP Model
<br>
<br>
## <b>Overview and Background</b>
<br>
<br>
<p align="center">
<img src="https://github.com/Lwhieldon/AbstractTextSummarization/blob/main/notebooks+utils+data/AbstractiveTextSummarization.png?raw=true" height=200 />
</p>

Text summarization is a complex task for recurrent neural networks, particularly in neural language models. Despite it's complexity, text summarization offers the prospect for domain experts to significantly increase productivity and is used in enterprise-level capacities today to condense common domain knowledge, summarize complex corpus of text like contracts, and automatically generate content for use cases in social media, advertising, and more. In this project, I explore the use of large language models in the recurrent neural network framework using encoder-decoder transformers from scratch to condense dialogues between several people into a crisp summary, demonstrating abstract text summarization. Applications of this exercise are endless, but could be especially beneficial for summarizing long transcripts from meetings and so on.

Let's first look at the dataset we will use for training: Samsung transcript data. We will then go into the scoring parameters and demonstrate how we train the model. Lastly, we will then showcase our model's inference and discuss opportunities for future work and study use cases.

## <b>Data & Model Details</b>
<br>
<p align="center">
<img src="https://github.com/Lwhieldon/AbstractTextSummarization/blob/main/notebooks+utils+data/SAMsum_sample.png?raw=true" height=300 />
</p>
For our application, we'll use the SAMsum dataset, developed by Samsung, which consists of a collection of dialogues along with brief summaries. In an enterprise setting, these dialogues might represent the interactions between a customer and a support center personnel or a transcript representing individuals taking part in a meeting, so generating accurate summaries can help improve customer service, cut down on note taking, and detect common patterns among customer requests or meeting themes.

For this project, we leverage 🤗 Hugging Face's SAMsum dataset by leveraging the load_dataset library. This is beneficial as Hugging Face has already performed the task of cleansing and organizing the SAMsum dataset for us.

The dataset has 3 features:
 - Dialogue, which contains the dialogue text,
 - Summary containing the synopsis of the dialogue, and
 - id to uniquely identify each record.
 
 🤗 Hugging Face's dataset is made of 16,369 conversations distributed uniformly into 4 groups based on the number of utterances in conversations: 3-6, 7-12, 13-18, and 19-30. Each utterance contains the names of the speaker. Note also the data is split into the following subsets:

Data Splits
- train: 14,732 records
- validation: 818 records
- test: 819 records  
### <b>Pegasus-Samsum Model</b>

This model is a fine-tuned version of [google/pegasus-cnn_dailymail](https://huggingface.co/google/pegasus-cnn_dailymail) on the samsum dataset ().
It achieves the following results on the evaluation set:
- Loss: 1.4919
### <b>Training hyperparameters</b>

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- gradient_accumulation_steps: 16
- total_train_batch_size: 16
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 500
- num_epochs: 1
### <b>Training results</b>

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| 1.6776        | 0.54  | 500  | 1.4919          |

<br>
<br>


## <b>YouTube Presentation</b>

To support the submission of this project to UMBC's Data Science Program, class DATA606: Capstone in Data Science, here is the youtube containing presentation. 

<br>
<br>

## <b>Table of Contents</b>
```
AbstractTextSummarization
|__ notebooks-utils-data
|   |__ProjectCode.ipynb
|__ presentation
|   |__ Presentation.pdf
README.md
```

## References
- M. Omar, S. Choi, D. Nyang, and D. Mohaisen, “Robust natural language processing: Recent advances, challenges, and future directions,” ArXiv Prepr. ArXiv220100768, 2022.
- C.-C. Chiu et al., “State-of-the-art speech recognition with sequence-to-sequence models,” in 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2018, pp. 4774–4778.
- I. Sutskever, O. Vinyals, and Q. V. Le, “Sequence to sequence learning with neural networks,” Adv. Neural Inf. Process. Syst., vol. 27, 2014.
- A. Vaswani et al., “Attention is all you need,” Adv. Neural Inf. Process. Syst., vol. 30, 2017.
- E. Voita, “Sequence to Sequence (seq2seq) and Attention,” Sep. 15, 2022. https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html
- K. Cho, B. Van Merriënboer, D. Bahdanau, and Y. Bengio, “On the properties of neural machine translation: Encoder-decoder approaches,” ArXiv Prepr. ArXiv14091259, 2014.
- J. Uszkoreit, “Transformer: A Novel Neural Network Architecture for Language Understanding,” Google Research, Aug. 31, 2017. https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html
- J. Vig, “Visualizing attention in transformer-based language representation models,” ArXiv Prepr. ArXiv190402679, 2019.
- G. Lovisotto, N. Finnie, M. Munoz, C. K. Mummadi, and J. H. Metzen, “Give Me Your Attention: Dot-Product Attention Considered Harmful for Adversarial Patch Robustness,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 15234–15243.
- E. Voita, D. Talbot, F. Moiseev, R. Sennrich, and I. Titov, “Analyzing multi-head self-attention: Specialized heads do the heavy lifting, the rest can be pruned,” ArXiv Prepr. ArXiv190509418, 2019.
- J. Alammar, “Illustrated Transformers,” Jun. 27, 2018. https://jalammar.github.io/illustrated-transformer/
- G. Ke, D. He, and T.-Y. Liu, “Rethinking positional encoding in language pre-training,” ArXiv Prepr. ArXiv200615595, 2020.
- H. Zhang, I. Goodfellow, D. Metaxas, and A. Odena, “Self-attention generative adversarial networks,” in International conference on machine learning, 2019, pp. 7354–7363.
- H. Luo, S. Zhang, M. Lei, and L. Xie, “Simplified self-attention for transformer-based end-to-end speech recognition,” in 2021 IEEE Spoken Language Technology Workshop (SLT), 2021, pp. 75–81.
- K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, “Bleu: a method for automatic evaluation of machine translation,” in Proceedings of the 40th annual meeting of the Association for Computational Linguistics, 2002, pp. 311–318.
- J. Zhang, Y. Zhao, M. Saleh, and P. Liu, “Pegasus: Pre-training with extracted gap-sentences for abstractive summarization,” in International Conference on Machine Learning, 2020, pp. 11328–11339.
- C.-Y. Lin, “Rouge: A package for automatic evaluation of summaries,” in Text summarization branches out, 2004, pp. 74–81.
- L. Tunstall, L. von Werra, and T. Wolf, Natural language processing with transformers.  O’Reilly Media, Inc., 2022.
- B. Gliwa, I. Mochol, M. Biesek, and A. Wawer, “SAMSum corpus: A human-annotated dialogue dataset for abstractive summarization,” ArXiv Prepr. ArXiv191112237, 2019.
- A. See, P. J. Liu, and C. D. Manning, “Get To The Point: Summarization with Pointer-Generator Networks,” in Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Vancouver, Canada, Jul. 2017, pp. 1073–1083. doi: 10.18653/v1/P17-1099.
- S. Maity, A. Kharb, and A. Mukherjee, “Language use matters: Analysis of the linguistic structure of question texts can characterize answerability in quora,” in Proceedings of the International AAAI Conference on Web and Social Media, 2017, vol. 11, no. 1, pp. 612–615.
- T. Tsonkov, G. A. Lazarova, V. Zmiycharov, and I. Koychev, “A Comparative Study of Extractive and Abstractive Approaches for Automatic Text Summarization on Scientific Texts.,” in ERIS, 2021, pp. 29–34.

## Project Curation

Note also that notebooks were created in Google Collaboratory to take advantage of GPU throughput to speed the training of the Transformers.
<br>
<pre>
Contributors : <a href=https://github.com/Lwhieldon>Lee Whieldon</a>
</pre>

<pre>
Languages    : Python
Tools/IDE    : Google Colab, Visual Studio Code
Libraries    : Transformers 4.22.2, Pytorch 1.12.1+gpu, Datasets 2.4.0, Tokenizers 0.12.1
</pre>
<pre>
Assignment Submitted     : December 2022
</pre>

