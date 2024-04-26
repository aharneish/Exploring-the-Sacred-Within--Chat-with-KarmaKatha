# codes
This reposotory contains all the codes used for fine tuning and rag implementation of the models that are OpenAI's GPT2 model, Meta's llama-2 model, MistralAI's Mistral-v0.1 model.
This repo also contains the dataset along with the collected data files.
 Abstract
This thesis focuses on developing an innovative spiritual chatbot rooted in Indian spiritual
wisdom. It aims to refine the model's ability to interpret and share insights from sacred texts,
distinguishing itself from previous initiatives like Gita-GPT. By infusing fresh perspectives, the
project aims to make ancient philosophies more accessible and relevant in today's context,
bridging the gap between antiquity and modernity.
The project's vision centers on revitalizing ancient teachings to resonate with contemporary
sensibilities, envisioning them as guiding principles for today's seekers. It serves as a conduit for
breathing new life into these teachings, making them more accessible, relatable, and essential in
modern existence.
7
Table of contents
Acknowledgements 5
Abstract 7
Table of contents 8
1. Introduction 10
1.1 Problem Statement 10
1.2 Dissertation Structure 10
1.3 Transformers 11
1.4 LoRA - Low Rank Adaptation of Large Language Model 12
1.5 PEFT - Parameter Efficient Fine Tuning 12
2. Literature Review 13
2.1 Transformers 13
2.2 Generative Pre-trained Transformer 2 (GPT-2) 15
2.3 Large Language Model Meta AI 2 (Llama-2) 16
2.4 Mistal-7b 17
2.5 Low Rank Adaptation of Large Language Model 19
2.6 Parameter Efficient Fine-Tuning 20
3. Data 21
3.1 Data Collection 21
3.2 Data Processing 21
3.3 Question and Answer Generation 22
4.Methods 24
4.1 Fine Tuning 24
4.2 Supervised Fine Tuning 26
4.3 RAG 26
4.4 Langchain 29
4.5 Vector Databases 30
4.6 Facebook AI Similarity Search (FAISS) 31
4.7 Retrieval-Augmented Generation Assessment (RAGAS) 32
4.8 Metrics 32
5. Experimental Details 34
5.1 GPT 2 (fine tuning) 34
5.2 GPT 2 SFT 35
5.3 Llama 2 (finetuning) 35
5.4 Llama 2 SFT 36
5.5 MistralAI SFT 36
8
5.6 RAG implementation 37
6. Results and Observations 38
6.1 Results 38
6.2 Observations 38
7. Conclusions and Future Scope 41
7.1 Conclusion 41
7.2 Future Scope 41
8. References 43
9
1. Introduction
Alan Perlis's statement, "A year spent in artificial intelligence is enough to make one believe in
God," encapsulates the profound awe and wonder that emerges from delving into the
complexities of AI. Yet, amidst the rapid advancements of technology, there persists a longing
for ancient wisdom found in our timeless texts. These texts serve as guiding lights, offering
solace and perspective in the tumult of modern life.
In seeking solace, individuals turn to ancient scriptures, such as the Bhagavad Gita, the Tao Te
Ching, or the teachings of Stoicism. These texts provide profound insights into human existence,
offering a roadmap for navigating life's challenges and uncertainties. Just as AI reveals the
intricate workings of the universe, these ancient texts unveil the deeper mysteries of the human
psyche and the cosmos.
The juxtaposition of AI and ancient wisdom underscores humanity's perennial quest for
understanding and meaning. While AI may inspire awe with its computational prowess, the
wisdom distilled from ancient texts resonates on a spiritual level, offering a sanctuary for the
soul in an increasingly digital world. Thus, the convergence of modern technology and ancient
wisdom prompts contemplation, inviting individuals to explore the intersection of science and
spirituality in their journey towards enlightenment.
1.1 Problem Statement
In the endeavor to develop a transformative spiritual chatbot grounded in Indian spiritual
wisdom, the primary objective is to create a dependable and efficient tool capable of answering
inquiries based on sacred texts. The distinctive approach involves fine-tuning a model dedicated
to the domain-specific task of comprehending and elucidating the profound teachings embedded
in these texts. While acknowledging existing efforts like Gita-GPT, a Retrieval Augmentation
Generation (RAG) model leveraging GPT-3, the initiative diverges by emphasizing the
meticulous refinement of the model to enhance its proficiency and accuracy in interpreting and
making accessible the timeless insights from Indian spiritual literature. Through this innovative
approach, the aspiration is to elevate the relevance of these ancient teachings in the
contemporary context, fostering a deeper understanding and appreciation of their profound
significance.
1.2 Dissertation Structure
This thesis comprises eight chapters, each serving a distinct purpose in elucidating the
progression and outcomes of the research journey.
In the initial chapter, we delve into the Introduction, where we expound upon the motivation
driving this thesis, elucidate the rationale behind the chosen subject matter, provide an overview
10
of the methodologies employed, and delineate the challenges encountered and opportunities
seized throughout the investigative process. Furthermore, this chapter sets forth the project's
problem statement, framing the subsequent discourse.
Chapter two delves into the Literature Review, tracing the trajectory of our exploration from
inception to current understanding. Here, we traverse the landscape of Natural Language
Processing, delving into Deep Learning paradigms, encompassing Transformer architectures to
the implementation of Retrieval Augmentation Generation (RAG) for conversational AI,
meticulously fine-tuned on domain-specific datasets.
The third chapter is dedicated to Data, recognizing its pivotal role in this thesis and any research
endeavor. It navigates through the journey of data scarcity, collection, cleaning, preprocessing,
and augmentation, pivotal steps in enriching the dataset for robust analysis.
Methods are elucidated in chapter four, where we expound upon the techniques and
methodologies deployed in model development. Grounded in principles such as Parameter
Efficient Fine-Tuning PEFT and Low Rank Adaptation of Large Language Models (LoRA), this
chapter also offers insights into pre-training and fine-tuning methodologies, complemented by
illustrative code snippets. Additionally, the chapter delineates the utilization of RAG for
knowledge base querying.
Chapter five delineates the Experimental Details, providing a comprehensive account of
hyperparameters and training specifics, facilitating reproducibility and enhancing the refinement
of models by fellow researchers.
Results and Observations are shared in chapter six, shedding light on the empirical findings
gleaned from the project's execution.
Chapter seven encapsulates the Conclusions drawn from the research endeavor, highlighting the
contributions made and outlining avenues for future exploration and expansion.
Finally, chapter eight serves as the References section, underpinning the thesis with a robust
foundation of prior research and scholarly discourse, enriching the understanding of the subject
matter and guiding further inquiry.
In the following subsection we shall be giving a brief description of the methods used.
1.3 Transformers
Transformers are a type of neural network architecture developed primarily for natural language
processing (NLP) activities. Transformers outperform older versions at understanding the
relationships between words in a sentence, even if they are far apart. This makes them ideal for
tasks such as machine translation, text summarization, and question answering.
11
1.4 LoRA - Low Rank Adaptation of Large Language Model
LoRA, or Low-Rank Adaptation, is a method for more effectively training natural language
processing models. Although powerful, large language models (LLMs) require a lot of memory
to fine-tune for different applications due to their abundance of parameters.To address this,
LoRA augments the current model layers with tiny, trainable matrices. These matrices function
as "adaptors," adjusting only a small portion of the model's initial parameters to suit a new
task.This drastically lowers the amount of parameters that need to be trained, which speeds up
fine-tuning, uses less memory, and even allows LLMs to operate on less capable hardware.
1.5 PEFT - Parameter Efficient Fine Tuning
Parameter-Efficient Fine-Tuning is referred to as PEFT. It's a method for more effectively
training large language models (LLMs) in natural language processing. Large language models
(LLMs)are extremely computationally expensive NLP models with billions of parameters; yet,
fine-tuning (training the models on specific tasks) can greatly increase their power. PEFT
addresses this by fine-tuning only a portion of the parameters in the LLM that has already been
trained. By focusing on modifying a smaller set of parameters specifically for the new task, it
"freezes'' much of the initial set.When compared to conventional fine-tuning, this reduces the
computational cost and memory usage dramatically.When installing LLMs on devices with
constrained resources or in scenarios where training time is critical, PEFT is an excellent
method.
12
2. Literature Review
This chapter provides an overview of the previous works leading up to this project and gives an
insight into the current project and the job done till now. The sections mentioned below discuss,
the basic architecture of the transformers models and the various models used and also the
techniques used for the better inference of the models.
2.1 Transformers
The paper"Attention Is All You Need" presents the Transformer, a model architecture based
solely on attention mechanisms, which replaces the recurrent layers commonly used in sequence
transduction models [2]. The authors propose this new architecture as an alternative to the
complex recurrent or convolutional neural networks used in existing models. Self-attention is
used by the Transformer model to identify global dependencies between input and output. [2],
allowing for more parallelization and achieving state-of-the-art results in translation tasks.The
results section shows the Transformer model's exceptional performance on the WMT 2014
English-to-German and English-to-French translation tasks, setting new benchmarks for BLEU
grades [49]. Additionally, the model generalizes well to English constituency parsing,
outperforming previously reported models.The attention visualizations provided in the document
illustrate the behavior of the attention heads in capturing long-distance dependencies, anaphora
resolution, and sentence structure, demonstrating the interpretability of the model.
Natural language processing has undergone a substantial transformation thanks to the
Transformer neural network architecture. By enabling simultaneous input processing, they
provide an advantage over conventional recurrent neural networks that process inputs
sequentially. This leads to an increase in efficiency. The attention mechanism that allows the
network to concentrate on particular segments of the input sequence is the foundation of the
transformer design. With the use of this attention mechanism, the transformer can effectively
handle tasks like language translation and text summarization by capturing long-range
relationships between words in a sentence. The Transformer Encoder-Decoder model, which
includes an encoder and a decoder, is a well-liked transformer model. [2]. The encoder processes
the input sequence and generates a set of hidden representations that are fed into the decoder
alongside a target sequence to generate an output sequence. Transformers have found wide
application in diverse areas, including machine translation, text classification, question
answering, image recognition, and speech recognition. They have become an essential tool in
natural language processing and are expected to lead to more breakthroughs in the field as their
development continues.
13
Neural networks use a process called self-attention to compute input sequence representations by
paying attention to various locations within the sequence. It has been extensively used in
transformer architectures, which have shown outstanding outcomes for jobs involving natural
language processing.
Figure 2.1 Transformer Architecture [1]
The input sequence is converted into questions, keys, and values in order to compute
self-attention. These elements are then utilized to compute weights that represent the significance
of each location for each query. The weights are then used to generate a weighted total of the
data, which is the output of the self-attention layer. [2]. In contrast to conventional recurrent
neural networks, self-attention offers various benefits. It can process inputs concurrently and
identify long-range relationships between words in a sentence, which increases performance [2].
Moreover, it can adapt to the task at hand by focusing on different parts of the input sequence
[2]. The input for the transformer model in NLP is typically a sequence of word embeddings
representing the words in a sentence or document. These embeddings are learned through
unsupervised learning techniques such as Word2Vec or GloVe [3,4] .The input sequence is then
passed through an embedding layer, which maps each word to a high-dimensional vector
representation. This layer is often pre-trained on a large text corpus and fine-tuned for the
specific task. The encoded input sequence goes through the encoder, which comprises several
self-attention and feed forward layers [2]. The self - attention layers enable the model to focus on
different positions in the input sequence, while the feed forward layers transform the outputs of
14
the self-attention layers using activation functions and linear operations [2]. The encoded
sequence is given to the decoder to produce the output sequence. The decoder also has
self-attention and feedforward layers, and can attend to both the encoded input sequence and
the previous elements in the output sequence. Overall, the input sequence is processed by
self-attention and feedforward layers to generate the output sequence for natural language
processing tasks.
The input sequence is encoded by the encoder component in the Transformer model for natural
language processing. Its several layers of feedforward and self-attention neural networks allow
the model to recognize long-range relationships between words and pay attention to various
segments of the input sequence. [2]. The encoder's output represents the input for downstream
natural language processing activities. The output sequence is generated by the Transformer
model's decoder, which uses self-attention and feedforward neural networks. To generate the
output, it takes into account the encoded input sequence as well as the previous elements of the
output sequence.
Now we shall look into some of the encoder only models that were used for the endeavor,
namely GPT-2,Llama-2-7b,Mistral-7b. We shall first start with GPT-2 which was released by
OpenAI. Then we shall proceed to Llama-2-7b which was released by meta (formerly facebook)
then followed by Mistral -7b by MistrtalAI.
2.2 Generative Pre-trained Transformer 2 (GPT-2)
The paper discusses the unsupervised learning capabilities of language models, particularly
GPT-2 [6], when trained on a new dataset of web pages called WebText. The authors demonstrate
that language models can begin to learn tasks such as question answering, machine translation,
reading comprehension [6], and summarization without explicit supervision when trained on
WebText [6]. The WebText dataset is created by scraping web pages curated by humans,
resulting in a high-quality dataset of over 8 million documents. The language model used is
based on the Transformer architecture and shows promising performance on various tasks,
including question answering, reading comprehension, and translation. However, the
performance on summarization tasks is still rudimentary according to quantitative metrics. The
authors also highlight the model's ability to generate news articles and translate between
languages. The document discusses the model's memorization behavior and provides samples of
completions from the smallest and largest models on random unseen WebText test set articles.
The scientists agree that the zero-shot performance of GPT-2 is still far from practical
implementations. However, the findings provide a promising approach towards constructing
language processing systems that can learn to do tasks directly without the requirement for
supervised adaptation or modification [6].
The work presented in the document is related to previous research on language models,
pre-training methods for language tasks, and alternative approaches to filtering and constructing
15
large text corpora of web pages [6]. The authors also discuss the performance of larger language
models trained on larger datasets [6] and compare their findings to previous work. The document
provides a comprehensive overview of the model's performance on various tasks, including
question answering, reading comprehension, translation, and summarization, and compares the
model's performance to supervised baselines. The authors also discuss the limitations of the
model's zero-shot performance and emphasize the need for further research to explore the
potential of unsupervised task learning.
Figure 2.3 GPT-2 Architecture[7]
2.3 Large Language Model Meta AI 2 (Llama-2)
Llama 2-Chat models stand out from other open-source chat models due to several key features.
These models are finely tuned and optimized specifically for dialogue use cases, making them
exceptionally well-suited for conversational interactions [5]. With a parameter scale ranging
from 7 billion to 70 billion, Llama 2-Chat models offer the capacity for generating more complex
and nuanced responses. Their superior performance on various benchmarks highlights their
effectiveness in producing high-quality dialogue. Moreover, the implementation of safety
enhancements, such as safety-specific data annotation, tuning, red-teaming, and iterative
evaluations, ensures responsible development and usage of these models. Notably, Llama 2-Chat
models exhibit competitiveness with certain proprietary closed-source models, underscoring their
quality and efficacy in dialogue tasks. The commitment to ongoing transparency, safety
16
improvements, and future enhancements further solidifies the position of Llama 2-Chat models
as advanced and evolving tools for dialogue applications [5].
Based on human evaluations, the Llama 2 models, particularly the Llama 2-Chat variants, have
shown strong performance in terms of both helpfulness and safety [5]. In terms of helpfulness,
the largest Llama 2-Chat model demonstrated competitiveness with ChatGPT, achieving a win
rate of 36% and a tie rate of 31.5% in comparisons across approximately 4,000 prompts with
three raters per prompt. These models also outperformed other chat models significantly on the
evaluated prompt set [5]. Regarding safety, evaluations conducted on around 2,000 adversarial
prompts by human raters indicated that the Llama 2-Chat models performed well in meeting
safety standards, likely attributed to the safety-specific measures integrated into their
development process. Overall, the human evaluations highlight the effectiveness of Llama
2-Chat models in delivering helpful and safe dialogue interactions, positioning them as
promising options for applications where these qualities are paramount [5].
Llama 2, particularly the Llama 2-Chat models, emerges as a viable alternative to closed-source
chat models for dialogue applications due to several key factors [5]. Through human evaluations,
Llama 2-Chat models have showcased competitiveness with certain proprietary closed-source
models, demonstrating their capacity to deliver high-quality dialogue interactions. Their strong
performance in terms of helpfulness and safety further solidifies their suitability for dialogue
tasks, meeting essential criteria for effective communication. With parameter scales ranging from
7 billion to 70 billion, Llama 2-Chat models offer the capability to generate sophisticated
responses, comparable to some closed-source counterparts. The integration of safety measures
during their development underscores a commitment to responsible usage, aligning with industry
standards for dialogue applications. Moreover, the emphasis on transparency and open
accessibility of Llama 2 models fosters trust and collaboration within the research community,
enhancing their appeal as a reliable alternative to closed-source chat models for diverse dialogue
applications [5].
Now we shall look into another encoder only model by MIstralAi known as Mistral-7b
2.4 Mistal-7b
Mistral-7B is a cutting-edge 7-billion-parameter language model that excels in both performance
and efficiency within the realm of Natural Language Processing (NLP) [8]. This model
surpasses previous benchmarks set by models like Llama 2 and Llama 1, showcasing superior
results in reasoning, mathematics, and code generation tasks. Mistral 7B achieves this by
utilizing innovative attention mechanisms like grouped-query attention (GQA) and sliding
window attention (SWA) , which enhance performance while reducing computational
costs[45,46,47].
17
The introduction of Mistral-7B addresses the challenge of balancing model size with
computational efficiency in NLP advancements. By outperforming larger models like Llama-2
and Llama, Mistral 7B demonstrates that careful design choices can lead to high performance
without sacrificing efficiency in real-world applications [5]. The model's architecture, based on a
transformer framework, incorporates specific parameters and attention mechanisms to optimize
performance across various tasks.One notable aspect of Mistral 7B is its fine-tuned model,
Mistral-7B – Instruct, which showcases exceptional performance in chatbot tasks. This
fine-tuned version outperforms other models in chatbot arenas and demonstrates the adaptability
and versatility of Mistral-7B for a wide range of applications [8]. The release of Mistral 7B
under the Apache 2.0 license further emphasizes its commitment to open access and
collaboration within the AI community.
In conclusion, Mistral-7B represents a significant advancement in the field of NLP, offering a
powerful yet efficient language model that pushes the boundaries of performance in various
tasks. With its innovative attention mechanisms, fine-tuning capabilities, and commitment to
accessibility, Mistral-7B sets a new standard for future language models.Mistral 7B's
architectural details reveal its transformer-based design with specific parameters like dim,
n_layers, head_dim, hidden_dim, n_heads, n_kv_heads, window_size, context_len, and
vocab_size [8]. The model introduces innovations like Sliding Window Attention (SWA) to
handle longer sequences efficiently and Rolling Buffer Cache to optimize memory usage during
inference [46,47].
The model's performance is extensively evaluated across various benchmarks, including
commonsense reasoning, world knowledge, reading comprehension, mathematics, and code
tasks. Mistral 7B consistently outperforms Llama models in these evaluations, showcasing its
superiority in reasoning, code generation, and mathematics tasks.Moreover, Mistral 7B's
efficiency is highlighted through the computation of "equivalent model sizes" compared to
Llama 2 models [5]. The model achieves remarkable performance levels comparable to larger
models, demonstrating its ability to compress knowledge effectively while maintaining high
performance standards.The discussion on system prompts emphasizes the importance of
enforcing guardrails in AI generation for front-facing applications. Mistral 7B's system prompt
enables users to guide model outputs within specified constraints, promoting ethical and
responsible AI usage. Additionally, the model showcases content moderation capabilities through
self-reflection prompts, enabling accurate classification of acceptable content and filtering out
harmful or inappropriate material.
The following section discusses the training method of large sized (parameters) models; the
technique is known as LoRA or the Low-rank adaptation of large language model [9].
18
2.5 Low Rank Adaptation of Large Language Model
The motivation behind LoRA stems from the challenges associated with fine-tuning massive
language models like GPT-3, which can have billions of parameters. Fine-tuning all these
parameters for each task is computationally expensive and memory-intensive [9,10].LoRA
addresses this issue by immobilizing the pre-trained model weights and incorporating trainable
rank decomposition matrices into all layers of the Transformer architecture.
Over-parametrized models frequently have a low intrinsic dimension; the authors assume that the
change in weights during model adaptation also has a low "intrinsic rank" [42,43]. This
understanding led to the invention of LoRA, in which dense layers of a neural network are
indirectly trained by maximizing rank decomposition matrices of the change in weights during
adaptation [9]. LoRA improves storage and computation efficiency by freezing pre-trained
weights and focusing on optimizing smaller low-rank matrices [9].One of the primary benefits of
LoRA is the ability to share a pre-trained model and create several small LoRA modules for
various applications. By freezing the shared model and effectively switching jobs through the
replacement of matrices A and B, LoRA considerably reduces storage needs and task switching
overhead [9]. Furthermore, LoRA improves training efficiency and lowers the hardware barrier
by up to three times when utilizing adaptive optimizers, as it only optimizes the smaller low-rank
matrices rather than calculating gradients for all parameters [9].
The paper emphasizes that LoRA's linear structure enables the smooth integration of trainable
matrices with frozen weights during deployment, resulting in no inference latency compared to
fully fine-tuned models [9]. Furthermore, LoRA is compatible with various prior methods and
can be combined with techniques like prefix-tuning, showcasing its versatility in enhancing
model adaptation strategies [9].
Figure 2.2 Working of LoRA[9]
19
We shall now look into the method that makes the application of LoRA possible in the hugging
face transformers via a python package known as peft which is available on the pypi sources.
The peft library allows for many other fine tuning methods like sft or the supervised fine tuning.
2.6 Parameter Efficient Fine-Tuning
An in-depth exploration of Parameter-Efficient Fine-Tuning (PEFT) methods for pre-trained
language models (PLMs) in the field of Natural Language Processing (NLP) [11].It explores a
variety of research papers and techniques aimed at diminishing the quantity of fine-tuning
parameters and memory usage, all while preserving or even improving performance on
downstream tasks. [11].
One key focus of the paper is the evaluation of different PEFT methods on models like
RoBERTa, T5, and LLaMA [44,45,46]. The results illustrate that PEFT methods are capable of
significantly reducing the number of trainable parameters, leading to improved efficiency and
performance across various benchmarks [11]. The study also investigates the memory efficiency
and applications of PEFT methods in multi-task learning and cross-lingual transfer
scenarios[11].The document categorizes PEFT methods and conducts experiments to assess their
effectiveness, providing insights into the future directions of research in this domain. It covers a
wide range of approaches, including UniPELT, Autopeft, sparse structure search, AdaMix, and
SparseAdapter, among others. These methods aim to enhance the efficiency and effectiveness of
fine-tuning large language models for diverse NLP tasks.
Furthermore, the paper discusses the performance of different adapter variants in the
"Large-Sparse" setting, where network pruning techniques from SparseAdapter are applied to
various adapter variants. ProPETL introduces a prototype network for different layers and tasks
using binary masks, showcasing the versatility of PEFT methods across different PLMs and
datasets.The study also introduces innovative methods like LoRA, DyLoRA, AdaLoRA, and
Kernel-wise Adapter to fine-tune large neural networks by introducing trainable low-rank
matrices, dynamically adjusting ranks, and treating attention heads as independent kernel
estimators. These methods aim to optimize model calibration, reduce memory usage, and
enhance parameter efficiency in fine-tuning pretrained models for specific tasks.
Now we shall look into the Retrieval Augmented generation using which we develop our
chatbot.
.
20
3. Data
Humby's assertion that "Data is the new oil" holds true in today's digital landscape, where an
abundance of data exists across the internet, offering diverse applications when appropriately
formatted. However, not all data, whether sourced from the internet or private repositories, is
conducive to deriving patterns and insights. Therefore, the careful selection of data is paramount
to ensure the robustness and reliability of any model.
Within this chapter, we will delve into the methodologies employed for data collection, explore
the intricacies of data processing, and examine the subsequent actions taken with the collected
data.
3.1 Data Collection
The first difficulty that anyone who wants to develop/train a deep learning model is that of data.
A deep learning model requires large amounts of data to recognise patterns and then modify the
weights it has initialized through various techniques.
Given my problem statement I have collected various documents that are related to the ancient
scriptures, preferably direct english translations of the documents. The documents that were
collected were: Puranas, Vedas, The Upanishads, Sai Literature, The Arthashastra, Chanakya Niti
etc.1
A total of 171 documents were gathered from the above mentioned sources.
3.2 Data Processing
The documents were collected in pdf format, some of them had a scanned text document which
could not be used to train the model. Hence they were converted to text documents using the
following online resources 2
Also unnecessary spaces had to be removed.Some of the documents had a lot of empty spaces
when they were converted to text files. The removal of extra spaces can be done using the python
library called re which uses regular expressions.
2Links for the resources
https://pdfsimpli.com/ https://smallpdf.com/blog/pdf-to-text https://www.pdf24.org/en
1 Links for the sources (data collection)
https://vedpuran.net/ http://holybooks.com/ https://archive.org/ https://upanishads.org.in/
https://wbsl.gov.in/library.action
21
Figure 3.1
These were then saved and used for fine tuning the model.
3.3 Question and Answer Generation
Now the next task was to generate Question and Answer pairs. For this there can be two
approaches, one of them would be to manually create the question and answer pairs from the
documents by going through the documents. The other method is to pass the documents to a
LLM which will generate the question and answer pairs. The method that had been viable for the
process given the time, it had been decided to go with the later option. The LLM used was
GPT-3 and GPT-3.5 with the following list of prompts:
22
The following image shows the dataset
A total of 8390 questions and answers were generated.
The generated question-answer pairs were subsequently divided into 200 test pairs and the
remaining pairs were employed for additional model fine-tuning.
23
4.Methods
In this chapter we are going to discuss the methods with which we fine tuned the models and
how we performed Retrieval Augmented Generation (RAG) and also discuss what are vector
databases and one of the open source vector databases that we have used and then discuss how
we have evaluated the responses which we have gathered through RAG.
Let us start with Fine Tuning.
4.1 Fine Tuning
Large Language Models (LLMs) have transformed Natural Language Processing (NLP) by
mastering intricate language patterns from vast datasets, revolutionizing the field. However, their
general-purpose nature often necessitates further specialization for optimal performance on
specific tasks. Fine-tuning emerges as a crucial technique to address this need, tailoring LLMs to
excel in a particular domain and unlock their full potential.
The Foundation of Pre-Training: LLMs undergo a rigorous pre-training phase, ingesting vast
quantities of text and code data. This process equips them with a robust understanding of
fundamental language relationships. While powerful, this foundational knowledge needs
refinement for tackling specific NLP applications like code generation, content creation in
diverse styles, or high-fidelity machine translation.
Fine-Tuning: Sharpening the Focus: Fine-tuning acts as a targeted training stage focused on a
specific task or domain. Imagine an LLM pre-trained on a comprehensive corpus of literature.
Fine-tuning for legal document generation entails exposing it to extensive legal text datasets,
allowing it to acquire the precise vocabulary, sentence structure, and legal referencing specific to
legal documents.
Methodologies for Effective Fine-Tuning:
Several established techniques contribute to successful LLM fine-tuning:
● Strategic Layer Freezing: During fine-tuning, the initial layers of the LLM, embodying
the core language understanding, can be frozen. This focuses the training process on
subsequent layers, allowing the model to adapt to the target task while preserving its
fundamental knowledge base.
● Leveraging Transfer Learning: The pre-trained weights from the LLM serve as a
launching pad for fine-tuning. These weights provide valuable initial positions,
facilitating the model to grasp task specifics more efficiently and with substantially less
data compared to training from scratch.
● Tailored Objective and Loss Functions:The selection of objective and loss functions
holds paramount importance in guiding the learning process. These functions are tailored
during fine-tuning to ensure the model optimizes its outputs for the particular task at
24
hand, whether it involves generating precise summaries, translating languages smoothly,
or composing various creative content formats.
Advantages of Fine-Tuning LLMs:
● Enhanced Task Performance: Fine-tuned LLMs demonstrably achieve state-of-the-art
performance on specific NLP tasks when compared to general-purpose models.
● Reduced Training Time: Transfer learning from pre-training significantly accelerates
training time compared to building a model entirely from scratch.
● Versatility through Adaptability: LLMs' ability to be fine-tuned for diverse tasks
underscores their versatility as powerful NLP tools across numerous applications.
Considerations for Fine-Tuning:
● Data Acquisition Hurdles: FFine-tuning frequently requires significant quantities of
high-quality, domain-specific data to achieve optimal outcomes.
● Overfitting: A Possible Concern: Overly enthusiastic fine-tuning on restricted datasets
can result in overfitting, where the model excels on the training data but struggles when
confronted with new data. Diligent construction of training and validation sets is essential
to alleviate this potential issue.
● Computational Resource Requirements: Training large-scale LLMs, particularly with
fine-tuning, necessitates significant computational resources like powerful GPUs and
specialized hardware.
Real-World Applications of Fine-Tuned LLMs:
Fine-tuned LLMs find application in a wide spectrum of NLP tasks, including:
● Advanced Text Generation: Composing creative content in diverse styles, including
poems, scripts, emails, or even code.
● High-Fidelity Machine Translation: Translating languages while preserving nuance and
context with high accuracy.
● Intelligent Question Answering Systems: Building chatbots or virtual assistants capable
of providing informative and comprehensive answers to user queries.
● Concise Text Summarization: Effectively condensing substantial amounts of text into
concise summaries.
● Accurate Sentiment Analysis: Understanding the emotional undertones and opinions
expressed in textual content.
The Evolving Landscape of Fine-Tuning:
With continuous advancements in LLM pre-training and increasing data accessibility, fine-tuning
is poised to become an even more powerful tool. Research is actively pursuing solutions to data
25
efficiency challenges and further refining fine-tuning techniques, expanding its reach and impact
across various NLP domains.
In the next section we shall be discussing one of the fine tuning methods known as Supervised
Fine Tuning also called as SFT which is an effective way to train a chat model i.e, a model which
can be made to answer like a human would.
4.2 Supervised Fine Tuning
SFT [11] is a technique to finetune a large language model (LLM) on a labeled data to do a
specific task such as question answering or classification[11]. In this method the LLM
understands the pattern like given an input how the output should be hence this is very useful for
doing RLHF (reinforcement learning with human feedback) which improves the model
generation and reduces the irrelevancy of the generated output of the LLM. The model below has
been first fine tuned on the collected data as mentioned in the Data section then it was further
fine tuned for the downstream task of question answering.
In the following section we shall discuss what is ands how we employed RAG in our endeavor to
develop a chatbot based on Ancient Indian Scriptures
4.3 RAG
Large language models (LLMs) demonstrate impressive feats of engineering, adept at text
generation, language translation, and question answering. However, they have limitations. Their
knowledge is static, based on their training data, which might not always be current.
Additionally, LLMs can struggle with tasks requiring factual accuracy.An innovative method
known as Retrieval-Augmented Generation (RAG) merges the advantages of retrieval-based and
generation-based models for knowledge-intensive natural language processing (NLP) tasks.[12].
The RAG model incorporates a retriever component that retrieves relevant passages from a
knowledge source, such as a large text corpus or a structured knowledge base [12], and a
generator component that generates the final output based on the retrieved information.
One key aspect of the RAG model is its ability to leverage both parametric and non-parametric
memory to complete tasks. The parametric knowledge stored in the model's parameters allows it
to generate accurate outputs without relying on specific documents. This is demonstrated through
examples where the model completes titles based on partial decodings, showing that specific
knowledge is stored in the model's parameters.The RAG model is evaluated on the FEVER
dataset for fact verification tasks, showing competitive performance compared to
state-of-the-art[12] models with complex pipeline systems. The RAG model achieves results
within 4.3% of these models without requiring intermediate retrieval supervision, highlighting its
effectiveness in knowledge-intensive tasks.Furthermore, the RAG model is compared to other
text generation approaches, such as retrieve-edit-rerank text generation [47], which focuses on
26
improving text generation by retrieving relevant information before generating the output. The
paper also explores the utilization of GPUs for billion-scale similarity search and introduces
TriviaQA, an extensive challenge dataset tailored for reading comprehension tasks. [47].
Figure 4.3.1 RAG Architecture [48]
Retrieval-Augmented Generation (RAG) tackles these issues by combining LLMs with external
knowledge sources. Here's a breakdown of how it works:
1. Information Retrieval: When presented with a prompt or question, RAG first taps into a vast
knowledge base, like Wikipedia. Using a retrieval system, RAG identifies documents relevant to
the prompt.
2. Context Injection: The retrieved documents are then fed alongside the original prompt to the
LLM. This injects relevant factual information into the LLM's processing.
3. Enhanced Generation: With this enriched context, the LLM generates its response. Ideally,
the response is more accurate, informative, and grounded in real-world knowledge.
Benefits of RAG:
Improved Accuracy: By referencing external knowledge, RAG helps LLMs avoid factual
errors and biases present in their training data.
Up-to-Date Information: RAG allows LLMs to access and integrate the latest information,
keeping responses relevant.
27
Domain Specific Expertise: RAG can be fine-tuned for specific domains by using relevant
knowledge bases. Imagine a medical LLM consulting medical journals with RAG.
Reduced Training Needs: Updating an LLM's knowledge typically requires retraining. RAG
allows for efficient updates by simply changing the connected knowledge base.
Real-World Applications:
Question Answering Systems: The Retrieval Augmentation Generation (RAG) model can
empower intelligent chatbots and virtual assistants, delivering precise and current responses to
user inquiries.
Content Creation: Imagine a marketing tool that uses RAG to generate product descriptions
based on the latest industry trends.
Scientific Research: RAG can assist researchers by summarizing scientific papers or generating
hypotheses based on existing knowledge.
The limitations of Retrieval-Augmented Generation (RAG):
1. Precision and Recall:Naive RAG may suffer from low precision, leading to misaligned
retrieved chunks, and low recall, resulting in the failure to retrieve all relevant chunks.
2. Complexity in Retrieval:Relying on external knowledge sources for retrieval can introduce
inaccuracies if the retrieved information is incorrect, affecting the overall quality of responses.
3. Computational Intensity:The retrieval component of RAG involves searching through large
knowledge bases or the web, which can be computationally expensive and slow, impacting
system performance.
4. Privacy Concerns:Retrieving information from external sources raises privacy concerns,
especially when dealing with sensitive data. Adhering to privacy regulations may limit access to
certain sources.
5. Factual Accuracy vs. Creativity:RAG is based on factual accuracy and may struggle with
generating imaginative or fictional content, limiting its use in creative content generation.
6. Storage Requirements: Storing and managing the vast amount of data required for retrieval
and generation in RAG systems can be challenging and costly, especially in enterprise settings
with dynamic data updates.
28
7. Maintenance Overhead:Vector databases used in RAG can be costly and require significant
effort to maintain, particularly when adding new data that necessitates rerunning the entire
dataset for re-embedding.
8. Hallucination Risk:The accuracy of responses in RAG is dependent on the quality of training
data and techniques applied. If incorrect data is fed into the system via vector retrieval, it can
lead to poor outputs or hallucinations.
9. Scalability:Addressing the scalability challenge is crucial for wider adoption of RAG models,
as they must effectively manage growing volumes of data and user interactions while
maintaining performance efficiency.
10. Hybrid Models Integration: Integrating RAG with other AI techniques like reinforcement
learning for versatility and context-awareness requires careful design and optimization to ensure
seamless operation and training.
In the following section we shall discuss one of the packages which allows us to perform RAG.
It is known as Langchain. This package is available in python.
4.4 Langchain
LangChain is an innovative framework designed to empower developers in creating applications
that harness the power of language models for sophisticated reasoning and context-aware
interactions. This framework, available in Python and JavaScript, offers a comprehensive
ecosystem with key components like LangChain Libraries, Templates, LangServe, and
LangSmith [37,38].
LangChain Libraries serve as the backbone, providing interfaces and integrations for various
components, enabling developers to combine them into cohesive chains and agents. Templates
offer deployable reference architectures for tasks like building chatbots or analytical tools.
LangServe facilitates deploying LangChain projects as REST APIs, making them accessible and
scalable. LangSmith acts as a developer platform for debugging, testing, and monitoring chains
built on any LLM framework [37,38].
The framework simplifies the development process by guiding developers from writing
applications using libraries to referencing templates for guidance. LangSmith aids in inspecting,
testing, and monitoring chains to ensure continuous improvement, while LangServe streamlines
the transformation of chains into APIs for easy deployment [37,38].
LangChain's significance lies in its ability to streamline the creation of generative AI application
interfaces, particularly in connecting powerful LLMs like OpenAI's GPT-3.5 and GPT-4 to
external data sources for NLP applications. It addresses challenges in prompt creation, response
29
structuring, model switching, and memory limitations commonly encountered when working
with LLMs [38,39].
The framework offers solutions to these challenges through prompt templates that reduce
boilerplate text, output parser tools for structured responses, an LLM class for easy model
switching, and chat message history tools to address memory limitations. By providing a
standardized interface, LangChain enables developers to build a variety of LLM-powered
applications, from chatbots and code analysis tools to data augmentation and text classification
systems [39,40].
One of the most important parts of the RAG system is a vector database or the knowledge base.
In the following section we shall be discussing what is a vector database and how we have used
it to perform RAG.
4.5 Vector Databases
Vector databases are integral to contemporary AI applications, notably in Retrieval-Augmented
Generation (RAG) models. These databases adeptly store, organize, and index high-dimensional
vector data, offering crucial support for diverse generative AI scenarios. Unlike conventional
databases, vector databases represent data points through fixed-dimensional vectors clustered
according to similarity, rendering them ideal for RAG applications necessitating rapid and
low-latency queries for high-dimensional data.[36].
Key features of vector databases include efficient storage and retrieval capabilities, scalability to
handle evolving data requirements, optimized query performance for real-time applications [36],
dimensional flexibility allowing for varying dimensions per vector, integration with AI and ML
frameworks for seamless deployment, and robust security features to ensure data integrity [36].
In the context of RAG models, vector databases enhance the efficiency and accuracy of
generative AI workflows by efficiently storing, indexing, and retrieving documents during
inference. They play a critical role in applications like recommendation engines and chatbots by
ensuring speed, precision, and scalability essential for these systems. By leveraging vector
databases, organizations can refine their models by integrating specific data into general-purpose
models like IBM watsonx.ai’s Granite via a vector database [36].
Leading vector databases used for RAG include Milvus, an open-source, highly scalable
database designed for efficient similarity search with advanced indexing algorithms, and
Pinecone, a dependable database that enables ultra-fast vector searches for search,
recommendation, and detection applications. These databases include user-friendly SDKs for
quickly developing large-scale similarity search services, a cloud-native architecture for flexible
scaling, support for a variety of data formats, and increased vector search functionality [36].
30
There are few popular open source and cloud based vector databases like chromadb, pinecone,
FAISS, Qdrant, weaviate to name a few. We have used the vector database FAISS by facebook
because of the speed at which it can retrieve the relevant documents and the ability to run on
both cpu and gpu.
In the following section we shall be looking at FAISS.
4.6 Facebook AI Similarity Search (FAISS)
FAISS[13] is a powerful library designed for efficient similarity search and clustering of dense
vectors[13]. It is particularly useful in scenarios where large-scale vector databases need to be
searched quickly to find the most similar vectors to a given query vector. FAISS utilizes
algorithms that can handle sets of vectors of any size, even those that do not fit in RAM, making
it ideal for scenarios with massive datasets [28,29].
In the context of Retrieval-Augmented Generation (RAG), FAISS plays a crucial role in
enhancing the performance of language models by enabling efficient retrieval of relevant
information from a vector database. RAG leverages FAISS to index and search through vectors
representing textual data, allowing for quick and accurate retrieval of contextually relevant
information during the generation process [28,31].
The integration of FAISS in RAG workflows involves several key steps. First, the vector
database is created using FAISS, where vectors representing textual information are indexed for
fast retrieval. These vectors are typically encoded using models like Sentence Transformers to
capture semantic similarities between text snippets. FAISS offers various indexing methods, such
as IndexFlatL2, which performs brute-force L2 distance searches, suitable for datasets where the
number of indexed vectors is manageable [29,31].
By utilizing FAISS within RAG pipelines, developers can enhance the efficiency and accuracy of
generative AI models by enabling them to retrieve relevant information from large datasets
quickly. This integration ensures that RAG models can access a vast amount of contextual
information stored in vector databases, leading to more informed and contextually relevant
responses generated by the AI system [28,31].
For every product or experiment we need to verify the results of the experiment with some
metrics. For RAG based systems we have a set of predefined metrics. To measure the metrics we
use a package known as RAGAS which helps us to measure the metrics with which we can
select our model which performs the best.
31
4.7 Retrieval-Augmented Generation Assessment (RAGAS)
RAGAS, which stands for Retrieval-Augmented Generation Assessment, is a comprehensive
framework designed to evaluate the performance of Retrieval-Augmented Generation (RAG)[18]
pipelines. It offers a set of metrics and tools that enable developers to assess the effectiveness
and quality of RAG applications without relying on human-annotated ground truth labels,
making it a valuable resource for enhancing the evaluation process of generative AI models like
RAG [17,18,19].
One key aspect of RAGAS is its reference-free evaluation approach, where it leverages Large
Language Models (LLMs) to conduct evaluations without the need for human-annotated data.
This methodology allows for more cost-effective evaluations and provides a more accurate
assessment of model performance by utilizing LLMs to compare generated answers against
given contexts and prompts [17,18].
The framework focuses on evaluating RAG pipelines at a component level, considering both the
Retriever component responsible for retrieving additional context from external databases and
the Generator component that generates answers based on augmented prompts. By evaluating
these components separately and together, developers can gain insights into the strengths and
weaknesses of their RAG pipelines, enabling them to identify areas for improvement and
optimization [17,18].
RAGAS introduces several evaluation metrics that contribute to the overall RAGAS score,
including context relevance, answer faithfulness, answer relevancy, and context recall. These
metrics provide a comprehensive view of how well a RAG pipeline performs in terms of factual
consistency, answer pertinence, precision in presenting relevant information from contexts, and
adherence to ground truth answers. By utilizing these metrics, developers can quantitatively
assess the performance of their RAG applications and track improvements over time[17,18]. We
shall be looking into those in the following section.
4.8 Metrics
RAGAS (Retrieval-Augmented Generation Assessment) introduces a set of metrics tailored to
evaluate the performance of Retrieval-Augmented Generation (RAG) pipelines, focusing on key
aspects like answer relevancy, context relevance, relevancy, and faithfulness. These metrics play
a crucial role in assessing the effectiveness and quality of RAG applications, providing valuable
insights into the model's performance across different dimensions [18,19,21].
1. Answer Relevancy:
Definition: Answer relevancy metric measures how pertinent and appropriate the generated
answer is in response to the given query.
32
Scoring: It is judged on a scale from 0 to 1, where higher scores indicate a more relevant and
accurate answer aligned with the query.
Example: A high answer relevancy score signifies that the model's response directly addresses
the query, while a low score indicates a less relevant or off-topic answer.
2. Context Relevance:
Definition: Context relevancy metric evaluates the relevance of the retrieved contexts from
external knowledge sources used to answer the query.
Scoring: Context relevance score provides insights into how well the retrieved information
aligns with the query and contributes to generating accurate responses.
Example: A high context relevancy score indicates that the retrieved contexts are closely
related to the query, enhancing the model's ability to generate contextually relevant answers.
3. Relevancy:
Definition: The relevancy metric assesses the overall relevance and significance of the
generated responses in the context of the query and retrieved information.
Scoring: It offers a comprehensive view of how well the model's outputs align with the input
query and the retrieved contexts, reflecting the model's ability to provide meaningful and
accurate responses.
Example: A high relevance score signifies that the model consistently produces relevant and
meaningful answers, enhancing the user experience and utility of the RAG application.
4. Faithfulness:
Definition: Faithfulness metric measures the accuracy and fidelity of the generated answers
compared to the ground truth or expected responses.
Scoring: It evaluates how well the model's outputs align with the correct answers, indicating
the model's ability to generate faithful and factually accurate responses.
Example: A high faithfulness score indicates that the model's answers closely match the
ground truth, demonstrating the model's reliability and accuracy in generating responses.
In summary, RAGAS metrics like answer relevancy, context relevance, relevancy, and
faithfulness provide a comprehensive framework for evaluating the performance of RAG
pipelines. These metrics offer valuable insights into the relevance, accuracy, and quality of the
generated responses, enabling developers to assess and enhance the effectiveness of their RAG
applications across various dimensions [18,19,21].
33
5. Experimental Details
In this chapter we shall be seeing how we trained the various models discussed in the previous
chapters and see how RAG was implemented and in the following chapter we shall see the
evaluation metrics on the test data.
5.1 GPT 2 (fine tuning)
Building upon the preceding subtopic, it's crucial to note that the model underwent a meticulous
fine-tuning process utilizing textual documents. Notably, this fine-tuning endeavor was
conducted without LoRA configuration, ensuring optimization in full precision. The corpus of
textual documents encompassed a total of 174 entries, each meticulously curated to enrich the
model's understanding. The fine-tuning process extended over 15 epochs, during which the
model dynamically adjusted its parameters to enhance performance. The learning rate for this
training phase was set at 1e-5, a carefully chosen value to facilitate effective convergence and
mitigate the risk of overfitting. Both the training and evaluation phases were executed with a
batch size of 16, ensuring efficient utilization of computational resources while maintaining a
balance between model complexity and training stability. This rigorous fine-tuning regimen
aimed not only to optimize the model's predictive capabilities but also to imbue it with a nuanced
understanding of the underlying textual nuances, thereby enhancing its efficacy in real-world
applications.
Figure 5.1.1 training hyper parameters using trainer api
34
5.2 GPT 2 SFT
The model underwent supervised fine-tuning on the aforementioned text data using a dataset
comprising 8.19k question and answer pairs. Fine-tuning was conducted with a specific focus on
adapting the model to question answering tasks. The LoRA configuration employed for
fine-tuning was as follows:
Figure 5.2.1 LoRA config
- r=16 (indicating the number of attention layers),
- lora_alpha=32 (representing the sparsity regularization strength),
- lora_dropout=0.05 (specifying the dropout rate),
- bias="none" (indicating no bias was incorporated),
- task_type="CAUSAL_LM" (denoting a causal language modeling task).
During the fine-tuning process, the model was trained for 300 epochs, allowing for extensive
adjustment of its parameters to optimize performance on question answering tasks. Both training
and evaluation utilized batch sizes of 32, facilitating efficient processing of the dataset while
ensuring stability during training.
Fine-tuning with a focus on question answering tasks enables the model to better understand the
context of questions and generate relevant responses. The utilization of LoRA configuration,
along with the specified task type, enhances the model's ability to capture causal relationships
within the text, thereby improving its performance in generating accurate answers to posed
questions.
were 32.
5.3 Llama 2 (finetuning)
The model underwent training using text documents, employing the LoRA configuration
parameters set as follows: r=4 (representing the number of attention layers), lora_alpha=32
(indicating the sparsity regularization strength), lora_dropout=0.05 (specifying the dropout rate),
35
and no bias was utilized. The training process continued until 5 epochs, at which point both the
training and validation loss stabilized.
Throughout the training phase, batch sizes for both training and evaluation were set to 2. This
smaller batch size allows for more frequent updates to the model's weights, enhancing the
convergence speed and potentially improving generalization performance. By monitoring the
loss curves during training, it ensures that the model doesn't overfit or underfit the data, reaching
a suitable level of optimization.
The utilization of LoRA configuration, combined with careful batch size selection and
monitoring of loss convergence, ensures that the trained model effectively captures the
underlying patterns within the text data, making it well-suited for various natural language
processing tasks.
5.4 Llama 2 SFT
The model underwent supervised fine-tuning (SFT) on the pre-existing model, following the
methodology outlined in the GPT-2 SFT model. The LoRA configuration parameters were set as
follows: r=4 (indicating the number of attention layers), lora_alpha=32 (denoting the strength of
the sparsity regularization), lora_dropout=0.05 (specifying the dropout rate), and no bias was
incorporated. The fine-tuning process spanned 30 epochs, utilizing a training and evaluation
batch size of 4.
Throughout the fine-tuning phase, adjustments were made to the model's weights to enhance its
performance on particular tasks or datasets, drawing upon the knowledge acquired from its
pre-training. This iterative procedure enables customization of the model's abilities to better cater
to the intricacies of the desired domain. Fine-tuning with a restricted number of epochs and small
batch sizes facilitates efficient adaptation while preventing overfitting to the training data. These
parameters collectively enhance the performance and adaptability of the fine-tuned model,
making it more adept for specialized tasks in natural language processing and associated
disciplines.
5.5 MistralAI SFT
A supervised fine-tuning process was conducted on the Mistral model using a dataset comprising
8.19k samples, structured with LoRA configuration parameters: rank 16, LoRA alpha 32, LoRA
dropout 0.05, and no bias. The dataset consists of questions paired with corresponding answers.
Fine-tuning aimed to enhance Mistral's performance by leveraging this specific dataset, refining
its understanding and response generation capabilities. The process involved iterative
adjustments to model parameters based on supervised learning principles, ultimately aiming to
optimize model performance for question-answer tasks.
36
5.6 RAG implementation
1. Prepare Data and Models:
Gather the necessary textual data for indexing and retrieval. Obtain the base and fine-tuned
versions of the Llama-2, MistralAi, and Zephyr models for RAG implementation.
2. Vectorization with LangChain:
Utilize LangChain to encode the textual data into dense vectors using appropriate language
models (e.g., Sentence Transformers). Create a vector database using LangChain Libraries to
store and index the encoded vectors efficiently.
3. Integrate FAISS for Similarity Search:
Incorporate FAISS into the RAG pipeline to enable fast and accurate similarity search of
vectors. Choose the appropriate FAISS indexing method (e.g., IndexFlatL2) based on the dataset
size and search requirements.
4. Implement RAG with LangChain and FAISS:
Develop the RAG pipeline that combines retrieval and generation components using
LangChain's templates and libraries.Configure the pipeline to retrieve relevant context using
FAISS[13] for each of the specified models (Llama-2, MistralAi, Zephyr).
5. Model Integration:
Integrate Llama-2 (chat model) Llama-2 (fine tuned) MistralAi (base model) MistralAi (fine
tuned) Zephyr-(alpha/beta) (base model) models into the RAG pipeline to handle retrieval and
generation tasks.
Figure 5.6 RAG pipeline
37
6. Results and Observations
6.1 Results
Parameters answer
relevance
Faithfulness Relevance
MistralAI 7b 0.4 0.6 0.6
Llama-2 (chat) 7b 0.64 1 0.9
Zephyr-alpha 7b 0.45 0.2 0.3
Zephyr-beta 7b 0.1 0.2 0.2
llama-2-(chat)
fine tuned
7b 0.85 1 1
MistralAI fine
tuned
7b 0.6 0.8 0.2
6.2 Observations
The results as shown in the above section shows that llama-2-chat and llama-2-chat-fine tuned
outperform the other models such as mistral ai and zephyr in terms of faithfulness and relevance.
The scores for the base models relating to the answer relevance is shown zero is due to an error
with the testing metrics package which is not able to parse the output to evaluate. The following
are the sample outputs of the models on a given test dataset:
38
Questions Llama-2-Chatbase-7b
Llama-2-ChatFine-Tuned-7b
Mistal-7b-FineTuned
Mistal-7b-base
Describe the story of
Lord Brahma's creation
of the universe as
depicted in the
Brahmanda Purana.
According to
the Brahmanda
Purana, the
universe was
created by Lord
Brahma
through a
process of
divine
self-division.
Brahma divided
himself into
three parts:
Brahma,
Vishnu, and
Shiva. Brahma,
the creator, was
born inside a
golden egg
called ananda,
containing all
that would be
created in the
universe.
The Brahmanda
Purana narrates
the creation of
the universe by
Lord Brahma.
Initially, only the
divine essence
existed, which
divided into
Brahma, Vishnu,
and Shiva.
Brahma, the
creator, emerged
from an egg
containing all
worlds, living
beings, and
celestial bodies.
In the
Brahmanda
Purana, Lord
Brahma creates
the universe at
the beginning of
the current cycle
of time (kalpa).
Born from a
golden egg,
Brahma, with
four faces,
brings forth all
worlds, celestial
bodies, and
living beings
through his
divine power
(svayam). The
present cycle is
known as the
Varaha kalpa,
possibly named
after Vishnu's
boar
incarnation.
In the Brahmanda
Purana, Lord
Brahma creates the
universe by
dividing himself
into three parts:
Brahma, Vishnu,
and Shiva.
Brahma, born from
a golden egg,
initiates creation
within the egg,
which contains all
worlds and living
beings. The
process involves
cycles of
destruction and
re-creation known
as pratisarga.
Name the cosmic egg
mentioned in the
Brahmanda Purana,
from which the
universe originates.
The cosmic egg
mentioned in
the Brahmanda
Purana, from
which the
universe
originates, is
called "An4a"
or "Cosmic
egg".
The cosmic egg
mentioned in the
Brahmanda
Purana is called
"An4a" or
"Cosmic egg".
The cosmic egg
mentioned in the
Brahmanda
Purana is called
"An4a" or
"Cosmic egg".
The cosmic egg
mentioned in the
Brahmanda
Purana is called
"An4a" or "Cosmic
egg".
Explain the concept of
Yugas (eras) as
depicted in the
Brahmanda Purana.
The Brahmanda
Purana
describes Yugas
as a cycle of
four stages:
Satya Yuga,
Treta Yuga,
Yugas in the
Brahmanda
Purana include
Satya, Treta,
Dwapar, and
Kali Yugas, each
characterized by
Yugas in the
Brahmanda
Purana consist
of Satya, Treta,
Dwapar, and
Kali Yugas,
each with
Yugas consist of
Satya, Treta,
Dwapar, and Kali
Yugas, each with
distinct
characteristics.
These cycles repeat
39
Dwapar Yuga,
and Kali Yuga,
each lasting
1,000 years,
with the cycle
repeating every
12,000 years.
specific
durations and
moral qualities,
spanning a total
of 12,000 divine
years.
specific
characteristics
and durations
totaling
4,320,000
human years.
every 12,000
years, with each
Yuga lasting for
specific durations.
Name the celestial
weapon associated with
Lord Vishnu in the
Brahmanda Purana.
The celestial
weapon
associated with
Lord Vishnu in
the Brahmanda
Purana is the
Brahmastra.
The celestial
weapon
associated with
Lord Vishnu in
the Brahmanda
Purana is the
Brahmastra.
The celestial
weapon
associated with
Lord Vishnu in
the Brahmanda
Purana is the
Brahmastra.
The celestial
weapon associated
with Lord Vishnu
in the Brahmanda
Purana is the
Brahmastra.
Explain the concept of
Srishti (creation) and
Pralaya (dissolution) as
depicted in the
Brahmanda Purana.
Srishti refers to
the creation
process by
Brahma, and
Pralaya refers
to the
dissolution
process where
the universe
returns to its
unmanifest
state.
Srishti refers to
the
manifestation of
the universe
from the
unmanifest state,
while Pralaya is
the dissolution
returning the
universe to its
original state.
Srishti is the
creation process
by Brahma,
while Pralaya is
the dissolution
process
returning the
universe to its
original state,
forming a
cyclical pattern.
Srishti involves the
manifestation of
the universe by
Brahma,
expanding from the
unmanifest to the
manifest state.
Pralaya,
conversely, is the
dissolution of the
manifest universe
back into the
unmanifest.
40
7. Conclusions and Future Scope
7.1 Conclusion
In conclusion, the development of a spiritual chatbot for the purpose of aiding seekers in their
quest for guidance has been explored extensively in this master's dissertation. Leveraging
cutting-edge technologies and innovative techniques, the project aims to offer a seamless and
insightful experience for users.
The incorporation of technologies such as transformers, GPT-2, LLAMA-2, LoRA, and
LangChain underscores a commitment to harnessing state-of-the-art tools in natural language
processing to enhance the capabilities of the spiritual chatbot.
Dataset creation utilizing chat GPT models, including versions 3 and 3.5, has laid the foundation
for the chatbot's knowledge base, ensuring a rich reservoir of responses to address the diverse
inquiries of users.
The integration of the Retrieval-Augmented Generation (RAG) technique enhances the chatbot's
ability to retrieve relevant information from a vast pool of knowledge sources and generate
coherent responses tailored to the user's query.
Furthermore, the incorporation of scoring mechanisms to evaluate answer relevancy,
faithfulness, and overall relevance enhances the chatbot's performance and reliability in
providing accurate and meaningful responses.
Highlighting the superior performance of models such as Mistral 7b and LLAMA-2-7b-chat
underscores the meticulous approach taken in selecting and fine-tuning the models to ensure
optimal performance in addressing spiritual inquiries.
7.2 Future Scope
The future trajectory of this project harbors a multitude of promising advancements poised to
enhance its efficacy and reach. Firstly, an imperative step involves implementing the capability
to execute the models on CPU by transforming them into gguf and ggml formats. This strategic
move not only broadens accessibility but also optimizes performance, enabling users to leverage
the models across diverse computing environments without being constrained by hardware
limitations. By extending compatibility to CPU execution, the project stands to cater to a wider
user base, fostering inclusivity and scalability in its deployment.
Furthermore, the project aims to bolster its dataset by integrating additional data spanning
diverse regions, religions, and languages. This augmentation strategy seeks to enrich the model's
41
understanding and representation of cultural nuances and linguistic variations, thereby enhancing
its adaptability and effectiveness across heterogeneous contexts. Incorporating such varied data
sources not only broadens the scope of applications but also fortifies the model's robustness and
generalization capabilities, ensuring its relevance and utility in addressing a myriad of real-world
scenarios.
Lastly, a pivotal step in expanding the project's accessibility entails deploying it on Hugging
Face, a renowned platform revered for its comprehensive model repository and user-friendly
interface. By leveraging the extensive reach and infrastructure offered by Hugging Face, the
project endeavors to transcend geographical boundaries and device constraints, making its
resources readily available to a global audience of researchers, developers, and enthusiasts. This
strategic deployment not only enhances the project's visibility but also fosters collaborative
innovation and knowledge-sharing within the burgeoning AI community, catalyzing
advancements and breakthroughs in natural language processing and related domains.
42
8. References
1. J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of Deep
Bidirectional Transformers for Language Understanding,” Oct. 11, 2018.
https://arxiv.org/abs/1810.04805
2. A. Vaswani et al., “Attention Is All You Need,” Jun. 12, 2017.
https://arxiv.org/abs/1706.03762
3. Mikolov, Tomas, et al. "Efficient estimation of word representations in vector space."
arXiv preprint arXiv:1301.3781 (2013). https://arxiv.org/abs/1301.3781
4. Pennington, Jeffrey, Richard Socher, and Christopher D. Manning. "Glove: Global
vectors for word representation." Proceedings of the 2014 conference on empirical
methods in natural language processing (EMNLP). 2014.
https://aclanthology.org/D14-1162.pdf
5. Touvron, Hugo, et al. "Llama 2: Open foundation and fine-tuned chat models." arXiv
preprint arXiv:2307.09288 (2023). https://arxiv.org/abs/2307.09288
6. Radford, Alec, et al. "Language models are unsupervised multitask learners." OpenAI
blog 1.8 (2019):9. Language Models are Unsupervised Multitask Learners
7. Tarján, Balázs & Fegyó, Tibor & Mihajlik, Péter. (2022). Morphology aware data
augmentation with neural language models for online hybrid ASR. Acta Linguistica
Academica. 69. 10.1556/2062.2022.00582.
8. Jiang, Albert Q., et al. "Mistral 7B." arXiv preprint arXiv:2310.06825 (2023).
https://arxiv.org/abs/2310.06825
9. Hu, Edward J., et al. "Lora: Low-rank adaptation of large language models." arXiv
preprint arXiv:2106.09685 (2021). https://arxiv.org/abs/2106.09685
10. Brown, Tom, et al. "Language models are few-shot learners." Advances in neural
information processing systems 33 (2020): 1877-1901. https://arxiv.org/abs/2005.14165
11. Xu, Lingling, et al. "Parameter-efficient fine-tuning methods for pretrained language
models: A critical review and assessment." arXiv preprint arXiv:2312.12148 (2023).
https://arxiv.org/abs/2312.12148
12. Lewis, Patrick, et al. "Retrieval-augmented generation for knowledge-intensive nlp
tasks." Advances in Neural Information Processing Systems 33 (2020): 9459-9474.
https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-A
bstract.html
13. FAISS https://faiss.ai/index.html
14. RAGAS https://docs.ragas.io/en/stable/
15. Langchain https://www.langchain.com/
16. https://towardsdatascience.com/beautifully-illustrated-nlp-models-from-rnn-to-transform
er-80d69faf2109
17. https://towardsdatascience.com/evaluating-rag-applications-with-ragas-81d67b0ee31a
18. https://docs.ragas.io/en/latest/concepts/metrics/index.html
19. https://deci.ai/blog/evaluating-rag-pipelines-using-langchain-and-ragas/
20. https://arize.com/blog/ragas-how-to-evaluate-rag-pipeline-phoenix
43
21. https://blog.langchain.dev/evaluating-rag-pipelines-with-ragas-langsmith/
22. https://medium.aiplanet.com/evaluate-rag-pipeline-using-ragas-fbdd8dd466c1
23. https://langfuse.com/guides/cookbook/evaluation_of_rag_with_ragas
24. https://aclanthology.org/2023.tacl-1.1.pdf
25. https://blog.demir.io/hands-on-with-rag-step-by-step-guide-to-integrating-retrieval-augme
nted-generation-in-llms-ac3cb075ab6f
26. https://towardsdatascience.com/how-to-build-a-semantic-search-engine-with-transformer
s-and-faiss-dcbea307a0e8
27. https://www.comet.com/site/blog/evaluating-rag-pipelines-with-ragas/
28. https://github.com/ZeusSama0001/RAG-chatbot
29. https://community.sap.com/t5/technology-blogs-by-members/vector-databases-and-embe
ddings-revolutionizing-ai-in-rag-in-llm-or-gpt/ba-p/13575985
30. https://www.linkedin.com/pulse/executive-guide-vector-databases-rag-models-david-h-de
ans-lfs6c
31. https://blog.gopenai.com/primer-on-vector-databases-and-retrieval-augmented-generation
-rag-using-langchain-pinecone-37a27fb10546
32. https://community.openai.com/t/best-vector-database-to-use-with-rag/615350
33. https://www.aporia.com/learn/best-vector-dbs-for-retrieval-augmented-generation-rag/
34. https://nanonets.com/blog/langchain/
35. https://www.techtarget.com/searchenterpriseai/definition/LangChain
36. https://www.datacamp.com/tutorial/introduction-to-lanchain-for-data-engineering-and-dat
a-applications
37. https://www.geeksforgeeks.org/introduction-to-langchain/
38. https://semaphoreci.com/blog/langchain
39. Chunyuan Li, Heerad Farkhoor, Rosanne Liu, and Jason Yosinski. Measuring the
Intrinsic Dimension of Objective Landscapes. arXiv:1804.08838 [cs, stat], April 2018a.
URL http: //arxiv.org/abs/1804.08838. arXiv: 1804.08838.
40. Armen Aghajanyan, Luke Zettlemoyer, and Sonal Gupta. Intrinsic Dimensionality
Explains the Effectiveness of Language Model Fine-Tuning. arXiv:2012.13255 [cs],
December 2020. URL http://arxiv.org/abs/2012.13255.
41. Liu, Yinhan, et al. "Roberta: A robustly optimized bert pretraining approach." arXiv
preprint arXiv:1907.11692 (2019). https://arxiv.org/abs/1907.11692
42. Raffel, Colin, et al. "Exploring the limits of transfer learning with a unified text-to-text
transformer." Journal of machine learning research 21.140 (2020): 1-67.
https://arxiv.org/abs/1910.10683
43. Simple and Effective Retrieve-Edit-Rerank Text Generation (Hossain et al., ACL 2020)
44. Ligot, Dominic. (2024). Performance, Skills, Ethics, Generative AI Adoption, and the
Philippines. 10.13140/RG.2.2.14759.52643.
45. Ainslie, Joshua, et al. "Gqa: Training generalized multi-query transformer models from
multi-head checkpoints." arXiv preprint arXiv:2305.13245 (2023).
46. Iz Beltagy, Matthew E Peters, and Arman Cohan. Longformer: The long-document
transformer. arXiv preprint arXiv:2004.05150, 2020.
44
47. Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences
with sparse transformers. arXiv preprint arXiv:1904.10509, 2019.
48. https://www.analyticsvidhya.com/blog/2023/10/rags-innovative-approach-to-unifying-ret
rieval-and-generation-in-nlp/
49. Thang Luong, Hieu Pham, and Christopher D. Manning. 2015. Effective Approaches to
Attention-based Neural Machine Translation. In Proceedings of the 2015 Conference on
Empirical Methods in Natural Language Processing, pages 1412–1421, Lisbon, Portugal.
Association for Computational Linguistics.
50. Peng, Yifan & Yan, Ke & Sandfort, Veit & Summers, Ronald & lu, Zhiyong. (2019). A
self-attention based deep learning method for lesion attribute detection from CT reports.
1-5. 10.1109/ICHI.2019.8904668.
51. Zhang, Zhihui & Fort, Josep & Mateu, Lluis. (2023). Exploring the Potential of Artificial
Intelligence as a Tool for Architectural Design: A Perception Study Using Gaudí’sWorks.
Buildings. 13. 1863. 10.3390/buildings13071863.
52. Benedetto, Irene & Koudounas, Alkis & Vaiani, Lorenzo & Pastor, Eliana & Cagliero,
Luca & Tarasconi, Francesco & Baralis, Elena. (2024). Boosting court judgment
prediction and explanation using legal entities. Artificial Intelligence and Law. 1-36.
10.1007/s10506-024-09397-8.
53. Paudyal, Bed. (2010). Imperial and National Translations of Jang Bahadur's Visit to
Europe. South Asian Review. 31. 165-185. 10.1080/02759527.2010.11932734.
54. Pathak, Avik & Shree, Om & Agarwal, Mallika & Sarkar, Shek & Tiwary, Anupam.
(2023). Performance Analysis of LoRA Finetuning Llama-2. 1-4.
10.1109/IEMENTech60402.2023.10423400.
