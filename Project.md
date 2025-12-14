> ![A close up of a logo Description automatically
> generated](media/image2.jpeg){width="2.583934820647419in"
> height="0.9491666666666667in"}
>
> **LD7083**

**Computing and Digital Technologies Project Dissertation**

> *Deep Learning Approach for Detecting AI-Generated and Manipulated
> Images*

by

> W23070772
>
> Vasavi Swetha Sappa
>
> Supervisor Name: Zainab Ibrahim
>
> A Dissertation
>
> Submitted to the Department of Computer and Information Sciences
>
> Northumbria University
>
> In Partial Fulfillment of the Requirements For the Master of Science
> in
>
> Computing and Technology
>
> submission date: 27/08/2025

# Declaration {#declaration .unnumbered}

> I declare the following:

1.  That the material contained in this dissertation is the end result
    of my own work and that due acknowledgement has been given in the
    bibliography and references to ALL sources, be they printed,
    electronic or personal.

2.  The Word Count of this Dissertation is 12,000.

3.  That unless this dissertation has been confirmed as confidential, I
    agree to an entire electronic copy or sections of the dissertation
    to being placed on Blackboard, if deemed appropriate, to allow
    future students the opportunity to see examples of past
    dissertations. I understand that if displayed on Blackboard it would
    be made available for no longer than five years and that students
    would be able to print off copies or download. The authorship would
    remain anonymous.

4.  I agree to my dissertation being submitted to a plagiarism detection
    service, where it will be stored in a database and compared against
    work submitted from this or any other Department or from other
    institutions using the service.

> In the event of the service detecting a high degree of similarity
> between content within the service this will be reported back to my
> supervisor and second marker, who may decide to undertake further
> investigation that may ultimately lead to disciplinary actions, should
> instances of plagiarism be detected.

5.  I have read the UNN/CEIS Policy Statement on Ethics in Research and
    Consultancy and I confirm that ethical issues have been considered,
    evaluated and appropriately addressed in this research.

> **Signature: Vasavi Swetha Sappa**
>
> **Date:**

**27/08/2025**

> **Declaration of the Use of AI tool**
>
> **EITHER**
>
> [✔]{.mark}I have not used AI at any point in preparing this
> assignment.
>
> **OR**

- I have used AI tools (including, but not limited to ChatGPT) to help
  me (select all that apply):

  - Generate initial ideas in response to the question.

  - Develop my structure.

  - Generate ideas for examples / sources.

  - Provide feedback and suggestions for improvement on my content.

  - Edit and improve my spelling and grammar.

  - Other: please explain:

> [✔]{.mark}I have NOT used AI to generate whole sentences, paragraphs,
> sections, or the whole of this assignment and I understand that this
> would be considered academic misconduct.

<table style="width:100%;">
<colgroup>
<col style="width: 23%" />
<col style="width: 44%" />
<col style="width: 9%" />
<col style="width: 22%" />
</colgroup>
<thead>
<tr class="header">
<th><blockquote>
<p><strong>Your signature:</strong></p>
</blockquote></th>
<th><p>Vasavi Swetha Sappa</p>
<blockquote>
<p>(electronic signature is sufficient)</p>
</blockquote></th>
<th><blockquote>
<p><strong>Date:</strong></p>
</blockquote></th>
<th>25/07/2025</th>
</tr>
</thead>
<tbody>
</tbody>
</table>

# Acknowledgements {#acknowledgements .unnumbered}

> Firstly, I would like to Thank God Almighty for giving me the strength
> and opportunity to complete this dissertation at prestigious
> Northumbria University, London. I would like to thank my Professor Dr.
> Zainab Ibrahim for guiding and supporting me throughout the
> dissertation. Our program leader Dr. Ning Tse made sure every module
> is clearly explained and conducted monthly meeting to clarify any sort
> of doubts regarding dissertation.
>
> My sincere gratitude to my parents without whom my Masters in
> Computing and Technology would not have started. My thanks to my
> brother Krishna who is a software tester at TCS, taught me how to
> integrate the pipeline.
>
> Finally, I would like to thank my classmates and friends who were
> present throughout my Masters, sharing and discussing ideas with them
> proved helpful.

[]{#Abstract .anchor}

# Abstract {#abstract .unnumbered}

> Fake images have adverse effects on society. They are used to spread
> fake news with deceptive images, facilitate criminals in crossing
> borders with forged passports, and enable malicious attackers to forge
> images for blackmail, sometimes leading to suicide of victims. Fake
> brand promotions involving celebrities or influencers mislead people
> into purchasing counterfeit products, including fake medicines, posing
> serious dangers. Therefore, research on detecting fake images is
> critically needed. This dissertation aims to detect fake images using
> a hybrid model combining a Siamese Network with ResNet-50 backbone for
> face morph detection and a Vision Transformer (ViT)-based network for
> global feature morph detection. Initially, the Siamese Network
> processes a pair of human images (real and morphed), capturing facial
> regions via YOLOv4-tiny, preprocessing them using a hash function, and
> extracting features through convolution, batch normalization, and ReLU
> activations. During testing, feature similarity is evaluated using
> Euclidean distance and advanced loss functions. If face morph
> detection fails, the input image is passed to the ViT model, which
> divides it into patches, embeds them with positional encodings, and
> processes them through transformer encoders with multi-head
> self-attention to detect artefacts. The proposed model effectively
> detects both local and global morphs with an estimated accuracy of
> 92.5%, showcasing its potential for robust fake image detection in
> real-world scenarios..

# Contents {#contents .unnumbered}

> **[Declaration\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--](#declaration)\--**
>
> **[Acknowledgements\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--](#acknowledgements)\--**
>
> **[Abstract\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--](#Abstract)\--**
>
> **[Contents\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--](#table-of-images)\--**

1.  [**Introduction\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--**](#introduction)
    **1 to 18**

    1.  [Aims](#aims)

    2.  [Background](#1.2_Background)

    3.  [Research Questions](#1.3_Research_Questions)

    4.  [Objectives](#objectives)

    5.  [Structure of the Report](#structure-of-the-report)

2.  **[Literature
    Review\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--](#literature-review)19
    to 26**

    1.  Existing Works on Morphed Image Detection

> 2.1.1 Literature Review on Face Comparison

2.  Existing Works on AI- Generated Image detection

<!-- -->

3.  [Research Methodologies](#_bookmark15) 27 to 38

    1.  Type of Research Design

    2.  Data Collection

    3.  Data Cleaning and Feature Extraction

    4.  Justification

    5.  Ethical Considerations

4.  **System Design 39 to 57**

    1.  Training and Testing

    2.  Potential Risk

    3.  Limitations

5.  [Results and Analysis](#_bookmark20) 58 to 66

    1.  Evaluation Metrics

    2.  Performance of Siamese Network

    3.  Performance of ViT

    4.  Performance of the Hybrid model

6.  [Evaluation and Implications](#_bookmark24) 67

    1.  Strengths and limitations

7.  [Conclusions and Recommendations](#_bookmark26) 68

    1.  Unanticipated Findings

7.2 Future Scope

[References](#references) 70 to 72

[Appendix A -- Ethical Approval](#appendix-a-ethical-approval) 74 to 80

[Appendix B -- AI Usage Declaration](#appendix-b-ai-usage-declaration)
81

> [Appendix C -- Meeting Logs](#appendix-c-meeting-logs) 82 to 90

# Table of Images {#table-of-images .unnumbered}

8.  **Fig 1:** Fake image of President Donald Trump shot dead

9.  **Fig 2:** Synthesizing a morph image(b) using facial features of
    subject 1&2

10. **Fig 3:** Image classification

11. **Fig 4:** Real Vs Fake Image

12. **Fig 5**: Morphed image created using morph tool.

13. **Fig:** GAN Architecture

14. **Fig 6:** Types of MAD (Morph Attack Detection)

<!-- -->

9.  **Fig 7:** Comparing Siamese Network with other networks

10. **Fig 8**: Visual Representation of data collection from different
    data sets

11. **Fig 9**: Preparing the Data

12. **Fig 10**: Demonstration of transfer learning

13. **Fig11**: Represents a table proving Transfer learning is efficient
    than training the model from scratch

14. **Fig 12**: Proposed Hybrid model architecture

15. **Fig13:** Work-flow of model based on input image

16. **Fig 14:** Proposed Siamese Network Structure (Source: Self-Created
    -Canva)

17. **Fig 15:** Impact of face extraction using Hash function A standard
    frame size of 200\*200 is obtained

18. **Fig 16 :** Locating AOI (Area of Interest) using YOLOv4-tiny

19. **Fig 17:** Calculating Euclidean Distance

20. **Fig 18:** Similarity Comparison Flow Chart

21. **Fig 19:** Proposed ViT (Vision Transformer) Architecture

22. **Fig 20:** Architecture of Transformer Network

23. **Fig 21:** Proposed Transformer Encoder Architecture

24. **Fig 22:** Representation of Query, Key and value networks of a
    self-attention head in encoder

25. **Fig 23**: Architecture of a perceptron

26. **[Fig 24]{.underline}**: The interface page displayed when model
    runs

27. **[Fig 25]{.underline}**: Prediction of fake /synthetic image by
    hybrid model

28. **Fig 26:** Prediction of real image by hybrid model

29. **Fig 27**: Work flow of proposed model for morphed input

30. **FIG 28:** Work flow of proposed model for deepfake input

31. **FIG 29:** Taken from journal

32. **Fig 30:** The accuracy curves and loss curves of various face
    recognition models

33. **Fig 31:** False positive rate vs true positive rate for ResNet-50
    network

34. **Fig 32:** Classification report of ViT on Deepfakes

35. **FIG 33:** Confusion Matrix of Vision transformer

36. **Fig 34:** ROC curve of hybrid model on train dataset

37. **Fig 35:** ROC -AUC of hybrid model on train dataset

38. **FIG 36:** Confusion matrix for test dataset

39. **FIG 37:** Confusion matrix on Train dataset

40. **FIG 38**:Bar graph to compare models performance on train and test
    dataset

# Introduction

> In today's digital era, detecting fake images has become increasingly
> important. With the widespread use of social media, people spend a
> significant amount of time-consuming visual content online. Therefore,
> it is our responsibility to prevent the spread of fake images on these
> platforms, as they can have harmful consequences on the society.
>
> For instance, scammers create edited nude images to blackmail
> teenagers, leading to severe psychological trauma and, in some cases,
> suicide. Fake images are also widely used in branding scams, where
> fraudsters manipulate celebrity photos to advertise counterfeit
> products, resulting in financial losses for innocent people,
> especially elderly individuals who are more vulnerable to such
> deception. Furthermore, fake certificate images are used to obtain
> false medical certifications, allowing unqualified individuals to sell
> harmful medicines and tablets, thereby endangering public health. Fake
> images are frequently used to spread misinformation and false news,
> which damages public trust and creates confusion in society. Creating
> such fake images does not require much technical skill, making it
> easier for malicious users to exploit these methods.
>
> ![A person holding a paper and a person lying on a television screen
> AI-generated content may be
> incorrect.](media/image3.png){width="3.8787882764654418in"
> height="2.1809765966754155in"}
>
> Fig 1:Fake image of President Donald Trump shot dead
>
> Source: (Sabitha et al., 2021)\[19\]
>
> An example of a manipulated image is shown in Fig. 1, which was
> circulated by a popular news site depicting Donald Trump being shot,
> accompanied by a "RIP" hashtag. This image gained over three million
> views and led to significant controversy.
>
> Criminals use forged passport images to cross international borders
> illegally, posing serious threats to national security. In several
> countries, passport applicants submit printed facial photographs,
> which are later scanned and digitally processed for passport
> production. As shown in Figure 2, face (b) combines the hair,
> forehead, and beard of person (a) with the eyes, nose, mustache, and
> lips of person (c). It is evident that Figure 2b bears a clear
> resemblance to both individuals depicted in Figures 2a and 2c. This
> system can be exploited by submitting an artificially generated facial
> image that visually and structurally resembles two or more
> individuals. (Scherhag et al., 2018) \[20\]
>
> ![A screenshot of a computer AI-generated content may be
> incorrect.](media/image4.png){width="6.064798775153106in"
> height="1.6060608048993876in"} Fig 2: Synthesizing a morph image(b)
> using facial features of subject 1&2
>
> Source: (Scherhag et al., 2018)

**Problem:**

Given these risks, ensuring the legitimacy, authenticity, and security
of digital images is essential. Significant research has been conducted
to address this issue. Earlier approaches involved Recurrent Neural
Networks (RNNs), which are effective for sequential data such as videos
but are not suitable for static image forgery detection.

## Aims

> []{#1.2_Background .anchor}The main aim of the project is to develop a
> hybrid model to detect fake images.
>
> 1.To implement a Siamese Network with ResNet50 as backbone that adopts
> separate filters to detect AIG and morphed images.
>
> 2\. To input images, capture facial regions, preprocess images using
> Hash function and then proceed to similarity comparison done by
> Siamese Network and Vision Transformer. (ViT)
>
> 3\. To improve the accuracy of the model by transfer learning method.
>
> []{#1.3_Research_Questions .anchor}**1.2 Background**
>
> A lot of illegal activities are taking place due to the misuse of
> image editing and synthesis. Although editing captivates attention in
> the entertainment industry, such as in movie graphics and VFX, it also
> has the potential to harm society. Understanding the disasters this
> technology could cause is important, whether it is the fake promotion
> of gambling apps by celebrities, which can lead common people to
> invest in betting and ruin their lives, or the endorsement of
> uncertified medicines that can harm health. Online romance scams are
> also on the rise, where attackers morph images of girls into nude
> photos to blackmail them. Even though we have cybercrime teams to
> handle such attackers, new techniques for creating or morphing images
> are evolving every day. Using watermarks to identify the genuineness
> of an image is no longer effective because there are tools that can
> erase watermarks or create fake ones. Similarly, specific access
> controls, like read-only documents for images, have failed since
> screenshots and screen recordings are easily available, allowing
> images to be edited. Encrypting images with passwords can help protect
> them, but it is nearly impossible to create and manage a password for
> every picture on the internet. Detecting morphed images with the naked
> eye is nearly impossible; only some obvious errors, like the wrong
> placement of eyes or nose on a face, can be detected directly by
> humans. Therefore, detecting complex fake images is only possible
> through models that learn the intrinsic patterns hidden in an image,
> such as artefacts and pixel relationships. Both the spatial and
> frequency domains of an image are studied in depth, and classification
> is performed accordingly
>
> **1.3 Research Questions**

##  1.3.1 What are fake images?  {#what-are-fake-images .unnumbered}

> There are two types of images namely real images and fake images. A
> real photo is an image made by capturing light on a sensitive surface,
> such as traditional film or a digital sensor, like those used in
> modern cameras or smartphones.

##  {#section .unnumbered}

Fig 3: Image classification

Source: Self Created(MS Word)

> Fake images include two categories Morphed and AI-generated images.
> Fake images are formed by manipulating the real images or using the
> real images as a reference various models can generate fake images

![A vase with pink flowers AI-generated content may be
incorrect.](media/image5.jpeg){width="1.7840277777777778in"
height="0.9090277777777778in"}![A vase with flowers on a table
AI-generated content may be
incorrect.](media/image6.jpeg){width="1.7215277777777778in"
height="1.0in"}

> Real Image AI-Generated Image

Fig 4: Real Vs Fake Image, Source: (Baraheem & Nguyen, 2023) \[11\]

##  {#section-1 .unnumbered}

## 1.3.2 What are the methods used to create morphed images and AI-Generated images? {#what-are-the-methods-used-to-create-morphed-images-and-ai-generated-images .unnumbered}

> Morphed images are mostly made by combining facial features of two or
> more existing persons images.

![A screenshot of a computer AI-generated content may be
incorrect.](media/image7.png){width="5.278219597550306in"
height="2.8179188538932634in"}

Fig 5: Morphed image created using morph tool. Source: (Morais et al.,
2025)\[22\]

> According to (Morais et al., 2025), The four main methods used for
> morphing are:
>
> 1\. Face retouching to change facial attributes such as age, skin
> tone, or gender.
>
> 2\. Face Swap to change the face Identity.
>
> 3\. Expression transferring.
>
> 4\. Entire face synthesis to generate fake face images that do not
> exist in reality.
>
> All of these morphs are mostly generated by a network called
> GAN(Generative Adversal Network)
>
> The GAN-based algorithms are employed to produce images from scratch
> and change the features of the existing photos \[1\]. A Generative
> Adversarial Network (GAN) consists of two main parts: a generator and
> a discriminator, which compete against each other during training. The
> generator's job is to create realistic-looking fake images in order to
> fool the discriminator, while the discriminator's task is to correctly
> identify whether an image is real or generated. In the initial stages
> of training, the generator produces images that look clearly fake,
> making it easy for the discriminator to classify them as fake and
> penalize the generator for its poor outputs. However, as training
> continues, the generator gradually learns to create more realistic
> images that can trick the discriminator. Eventually, the discriminator
> becomes unable to distinguish between real and generated images,
> resulting in the generation of high-quality, photo-realistic images
> (Baraheem & Nguyen, 2023).
>
> ![A diagram of a software system AI-generated content may be
> incorrect.](media/image8.png){width="5.916666666666667in"
> height="3.2152766841644795in"}
>
> Fig: GAN Architecture, Source: (Morais et al., 2025) \[22\]

Subsequently, the diffusion model, AI image-generation models utilizing
the transformer structure is used for generating fake images.

**1.3.3 What are the methods used to detect morphs?**

> Morph Attack Detection (MAD) is of two types: Differential Morphing
> Attack Detection (D-MAD) and Single-image Morphing Attack Detection
> (S-MAD).
>
> D-MAD compares a suspected morphed image with a trusted reference
> image,
>
> S-MAD, detects whether a given single facial image has been morphed
> without any reference image available for comparison. : (Morais et
> al., 2025) \[22\]. Artifact-feature-based detection performs well on
> GAN and real images, whereas image-encoder-feature-based detection
> performs well on diffusion and transformer images.
>
> There are two main types of forged image detection methods: intrusive
> and non-intrusive techniques. Intrusive methods, like digital
> signatures and digital watermarking, require adding some extra
> information to the image when it is created, which helps later in
> checking if the image has been morphed. On the other hand,
> non-intrusive methods do not need any such prior addition and work
> directly on the available image. These are further divided into
> dependent and independent techniques. Dependent methods, such as
> copy-move and image splicing detection, compare the image with
> original data or check for duplicated parts to find any tampering.

![A diagram of a method AI-generated content may be
incorrect.](media/image9.jpeg){width="4.825694444444444in"
height="3.5in"}

Fig 6: Types of MAD (Morph Attack Detection)Source: (Sabitha et al.,
2021) \[19\]

**1.3.4 What are the threats caused by fake images?**

> The rise of realistic fake media poses serious threats to legitimacy,
> authenticity, and security, as malicious users can exploit these
> technologies to create forged content that is difficult to detect
> (Baraheem & Nguyen, 2023) \[10\]. Scammers often use deepfake videos
> to impersonate celebrities or political figures, making it appear as
> if they endorsed certain products or made statements they never
> actually said, which tricks people into buying fake items, visiting
> fraudulent websites, or sharing their personal information (Puzder,
> 2024) \[23\]. Additionally, cybercriminals generate fake photographs
> for online dating profiles to deceive victims into giving away money
> or sensitive details, and they also create false images of disasters
> to collect donations that never reach real victims (Puzder, 2024)
> \[23\]. The spread of such fake videos and manipulated faces online
> creates various ethical, social, and security challenges, including
> the promotion of false news and enabling different types of fraud
> (Alabdan et al., 2024) \[11\]. Criminals can also set up fake websites
> and social media profiles to spread misinformation or advertise
> counterfeit products for financial gain (Baraheem & Nguyen, 2023)
> \[10\].

## Objectives  {#objectives}

Detecting both morphed images and AIG images is complicated and
challenging. Morphed images need face recognition technology to compare
the faces of two persons while AIG images need a global view of the
image to detect explicit artifacts in terms of pixel anomalies and
implicit artifacts, which are considered as artificial
signatures/fingerprints based on the generative model architecture. So
the proposed model is a combination of face recognition and artifacts
detection using a hybrid model.

The main objectives of the project to achieve the aim of identifying
fake images are constructed as follows,

> 1\. A Siamese network with ResNet50 as background. The Siamese network
> takes real and morphed face images as input and processes them through
> identical subnetworks to extract features. It then compares the
> resulting feature vectors using contrastive learning, making it
> effective for high-similarity tasks. (Zhang et al., 2024) .
>
> 2\. A ViT(Vision Transformer) takes a single image as input ,performs
> Patch Embedding-The image to be checked is divided into patches of
> size 16\*16 pixels, Positional Encoding-The process of storing spatial
> information in each patch ,this is done to assist model the position
> of the patch while reconstructing the image. Before feeding these
> embeddings to the transformer encoder, a special classification token
> is added near the first patch, often called \[CLS\]. Transformer
> Encoder is a neural network with several multi head self attention and
> feedforward heads to study long-range dependencies and interactions
> among various components of the image. The Classification Head passes
> the output through a fully connected layer and a softmax activation
> function to classify image as real or fake
>
> This \[CLS\] token does not represent any patch but is a learnable
> embedding designed to gather global information from the entire
> sequence after processing. (Lamichhane, 2025)\[13\]

## Structure of the Report

##  {#section-2 .unnumbered}

## The entire dissertation is structured into a report containing six chapters. The first chapter is the introduction, which is the section explained before. The upcoming chapter is Literature review, then Methodology, then System Design, followed by results and evaluation, discussion, conclusion, and future work. The precise structure is as follows: {#the-entire-dissertation-is-structured-into-a-report-containing-six-chapters.-the-first-chapter-is-the-introduction-which-is-the-section-explained-before.-the-upcoming-chapter-is-literature-review-then-methodology-then-system-design-followed-by-results-and-evaluation-discussion-conclusion-and-future-work.-the-precise-structure-is-as-follows .unnumbered}

##  {#section-3 .unnumbered}

> Chapter 2 - Literature Review
>
> Chapter 3 - Research Methodology and/or Design of Practical Work
>
> Chapter 4 - Results and Data Analysis
>
> Chapter 5 - Evaluation and Implications
>
> Chapter 6 - Conclusions and Recommendations

# 2 Literature Review {#literature-review .unnumbered}

Many researchers have worked on detecting fake images. Different methods
have been proposed to identify morphed images and AI-generated images
(AIGs). Some models are designed to detect only morphed images, while
others focus on AI-generated images, and a few are capable of detecting
both types. A detailed review of these existing models has provided
valuable insights for developing a robust approach to detect both
morphed and AI-generated images effectively.

**Current Approaches and Limitations**

Wehrli et al. (2022) assessed that CNNs, for instance, have demonstrated
orientation in recognizing specific facial structures within diverse
groups but perform poorly when called upon to differentiate relatively
similar faces. Furthermore, other techniques like hyperspectral imaging,
which is able to pick data beyond the range of normal vision, hold a lot
of potential, but they have not made their way into widespread usage
mostly because of high cost and technicality in implementation.

## Existing Works on Morphed Image Detection

- (Morais et al., 2025)\[1\], proposed learning pixel differences in
  image using features like Local Binary Patterns (LBP) and Binarized
  Statistical Image Features (BSIF), but their performance dropped with
  compression and varied datasets. This shows they're not flexible
  enough for real-world conditions. Then came CNN(Convolution Neural
  Network).

- (Venkatesh et al., 2020)\[2\], proposed a method based on
  Convolutional Neural Networks (CNN) to detect face morphing attacks by
  analyzing noise patterns in facial images. The key idea is that
  morphed images exhibit unusual or inconsistent noise compared to
  genuine images. These noise patterns act as important clues in
  identifying whether an image has been tampered with. To capture these
  subtle patterns, the image is first enhanced using several denoising
  techniques. The methods include Wavelet Denoising, Block Matching and
  3D Filtering (BM3D), Multiresolution Bilateral Filtering (MBF), and
  Denoising Convolutional Neural Network (DnCNN). The outputs of these
  methods are combined to extract a cleaner residual noise image.
  Feature extraction is then performed using AlexNet, a pre-trained CNN
  model known for its capability to identify discriminative patterns.
  The extracted features are classified using a Probabilistic
  Collaborative Representation Classifier (P-CRC), which helps in
  distinguishing morphed images from real ones. In addition to CNN
  features, the approach also incorporates texture descriptors such as
  BSIF, HOG, and LBP. These descriptors add further depth to the texture
  analysis, thereby boosting the overall performance of the morph
  detection system.

- (Qin et al., 2022) \[3\], proposed a Convolutional Neural Network
  (CNN). The proposed method uses loss function for morphing attack
  detection. Unlike traditional approaches that compute a single overall
  loss for the entire image and optimize the model accordingly, this
  method applies a feature-wise loss, where loss is calculated across
  different regions or features of the image. This allows the model to
  focus specifically on areas that exhibit abnormal patterns caused by
  morphing. By providing supervision at these localized regions, the
  network becomes more sensitive to subtle artifacts and distortions,
  improving its capability to distinguish between genuine and morphed
  facial image.

- (Medvedev et al., 2023) \[4\], proposed two parallel Resnet 50
  Convolution Neural Network model each network trains under different
  identity labels of morphed images to learn various aspects of identity
  features like eyes, nose, mouth and artifacts. Based on this training
  the model learns fake images. The outputs of both networks are
  combined using dot product and sigmoid function classifies image as
  real or fake.

- (Banerjee & Ross, 2021) \[5\], developed a Conditional Generative
  Adversarial Network (cGAN), which takes two inputs to detect morph.
  First input is a Bonafide /real image and the second is a morphed
  image. The model uses disentangling to remove the morphed features in
  second image and then reconstructs a reference image. A biometric
  comparator is then used to compare the real and the generated
  reference image. If the similarity score is low then the second image
  provided as input is said to be morphed.

- (Blasingame & Liu, 2021)\[6\] , proposed a model which synthesies
  images using a Generative Adversarial Network. Real images are given
  as input ,the model has encoder which converts images into a latent
  code. The latent code of any two images are combined by combinator to
  produce a new morphed image. This morphed image is passed through
  detector /critic network to learn the difference between the real
  image and the morphed image. Generator produces a new morph which the
  detector cannot identify. The detector learns the difference. This
  cycle continues and after certain training period, the detector/critic
  performs accurately and is now used as a model to identify morphed
  images.

- (Wang et al., 2023) \[18\], proposed a CNN-based deepfake detection
  approach, called Attention-expanded Deepfake Spotter (ADS). The model
  locates manipulated regions by training a modified CNN to produce
  Residual Similarity Maps (RSM). the EUR loss function, allows the CNN
  in the second stage to learn more discriminative features from
  manipulated areas that were missed during the first stage.
  Furthermore, an ensemble strategy to select and combine candidate
  models effectively was implemented to reduce detection errors caused
  by example forgetting.

- (Sabitha et al., 2021) \[19\], proposed a method to detect the images
  that are altered with splicing methods. Images from benchmark datasets
  CASIA v1.0 and Celeb A were used for training the model. Enhanced
  Model for Fake Image Detection (EMFID), has four phases, namely: Image
  Pre-Processing, Histogram based Feature Extraction, Discrete Wavelet
  Transform, Image Classification with Convolutional Neural Networks
  (CNN). 94.3% (in avg) accuracy with higher precision rate, reduced
  error with minimal processing time in forgery detection was observed.

After reviewing several journals on morphed face detection common
texture-based features are not effective for this problem, it is evident
that incorporating face comparison techniques can enhance detection
performance. This method works by calculating the differences between
the landmarks (Scherhag et al., 2018) \[15\]. Comparing specific facial
features such as the eyes, nose, lips, and overall face curvature
between original and morphed images can aid in accurately identifying
morphs. The following literature presents a detailed analysis of various
face comparison algorithms relevant to this area of study.

1.  **Literature Review on Face Comparison**

In the context of morphed images, feature extraction is a crucial step,
as their faces are very similar and small differences must be carefully
captured.

- (Mousavi et al., 2021) \[7\], utilizes two techniques to pinpoint
  unique facial areas in identical twins: a Modified SIFT (M-SIFT)
  algorithm implemented on segmented facial regions, and a crowdsourcing
  method that involved 120 participants. M-SIFT key points and facial
  landmarks were utilized to extract features, resulting in the lowest
  equal error rates of 7.8%, 8.1%, and 10.1% across various image sets.

- (Chandana.S & R, 2022) \[8\], This research evaluates the
  effectiveness of an advanced Convolutional Neural Network (CNN) and a
  Support Vector Machine (SVM) in recognizing identical twins. Utilizing
  two samples for each algorithm from a gathered dataset, the CNN
  attained a notably superior accuracy of 94.01% in contrast to SVM's
  83.98% (p = 0.029).

- (Hattab & Behloul, 2023) \[9\], proposed a robust multimodal biometric
  recognition system that fuses face and iris modalities.
  Regions of interest are detected using YOLOv4tiny, and features are extracted using a new, efficient Deep Learning model that was modelled after the Exception pre-trained model.
  Additionally,
  employed Principal Component Analysis to preserve the permanent features,
  dimensionality reduction and Linear SVC for classification. Various
  fusion strategies---image-level, feature-level, and score-level---are
  evaluated. Using two-fold cross-validation on the CASIA-ORL and
  SDUMLA-HMT datasets, the system achieved 100% accuracy, demonstrating
  high reliability and performance.

- In this approach, an image is first input into the system, where Gabor
  and Local Binary Pattern (LBP) algorithms are used to extract
  distinguishing features. These features are then compared, and a
  multi-class SVM classifier is applied to categorize the individuals.

## Existing Works on AI- Generated Image detection

- (Alabdan et al., 2024) \[10\], proposed the ODL-GANFDC technique,
  which includes a CLAHE-based contrast enhancement stage to improve
  image quality. Histogram equalization adjusts the contrast of an image
  by applying the enhancement uniformly across the entire image. This
  method can sometime lead to loss of local details. Using Contrast
  Limited Adaptive Histogram Equalization (CLAHE) which divides the
  image into several small, non-overlapping regions called tiles. It
  performs histogram equalization on each tile separately. This
  localized enhancement improves the visibility of fine details and
  enhances structural features without introducing excessive noise. The
  deep residual network (DRN) model is then used to learn complex and
  intrinsic patterns from preprocessed images. The hyperparameters of
  the DRN model can be optimally selected using an improved sand cat
  swarm optimization (ISCSO) algorithm. Additionally, GAN-generated
  faces can be detected using a variational autoencoder (VAE).

- (Baraheem & Nguyen, 2023)\[11\], Convolutional Neural Networks (CNNs)
  were used. Transfer learning was applied, and Class Activation Maps
  (CAM) were used to identify the discriminative regions that the
  classification model focused on for its decision. Pre-trained models,
  including VGG19, ResNet (50, 101, 152 layers), InceptionV3, Xception,
  DenseNet121, InceptionResNetV2, MixConv, MaxViT, and EfficientNetB4
  were fine-tuned, replaced their ImageNet-trained classification heads
  with a new head consisting of global average pooling, a dense ReLU
  layer, batch normalization, dropout, and a final sigmoid dense layer.
  They trained with a batch size of 64, an initial learning rate of
  0.001, and for 20 epochs, saving checkpoints when validation loss
  improved. Horizontal flip was used for data augmentation. Adam
  optimizer was applied for all models. ResNet101, used RMSprop
  optimizer.

- (Xu et al., 2024) \[12\], A global feature extraction module using an
  attention-based MobileViT (AT-MobileViT) is designed to learn deep
  representations of global trace information. Multiple enhanced
  residual blocks are applied to extract discriminative multi-scale
  features. After this, a low-level feature extraction module that
  includes a channel-spatial attention (CSA) block is used. To help
  capture complementary information between features, a dual-branch
  interactive feature fusion module is introduced, which reshapes
  feature vectors into interactive matrices.

- (Lamichhane, 2025) \[13\], introduced a Vision Transformer (ViT) model
  that showed higher accuracy compared to Convolutional Neural Networks
  (CNNs). The self-attention mechanism in ViT helps to capture the
  overall context of an image and manage variations in its content that
  CNNs might fail to detect when identifying AI-generated images. In
  this approach, images are treated as sequences of patches, and these
  patches are embedded linearly before being input into a transformer
  encoder. The model applies self-attention mechanisms to recognize
  patterns and features. Therefore, the ViT model can capture long-range
  dependencies and complex relationships between different image parts,
  making it suitable for tasks requiring detailed analysis. The model
  architecture included Patch Embedding, Positional Encoding,
  Transformer Encoder, and a Classification Head. The pretrained model
  reached an accuracy of 98.2%.

- (Morais et al., 2025) \[14\], Addressed performance drops issue when
  dealing with out-of-domain generators and image scenes. Existing
  methods experience a significant accuracy drop when tested on
  cross-domain samples because the learned feature spaces contain
  domain-biased information, often overfitting to specific generator
  artifacts. An Artifact Purification Network (APN) is proposed to
  extract and analyze domain-agnostic artifact features through two
  purification processes. The APN consists of explicit purification,
  implicit purification, and a classifier. The explicit purification
  module separates artifact features using two branches: one focuses on
  frequency-band proposals, and the other on spatial feature
  decomposition. The implicit purification module aligns features from
  both branches and further refines them using a mutual information
  estimation-based training strategy. Unlike explicit purification,
  implicit purification is applied only during training and is not used
  during inference. Finally, the averaged artifact-related features are
  passed to the classifier to produce the decision probability.

- ProGAN-generated images and latent-diffusion-generated images, these
  datasets were to train model to evaluate six methods of AI generated
  image detection \[16\].

1.  Artifact-feature-based detection identifies common artifacts in
    AI-generated images. A deep learning model using a pretrained
    ResNet50 was trained to classify real versus AI-generated images,
    evaluated on eleven CNN-based generative models. Probabilistic data
    augmentation methods like Gaussian blur and JPEG compression
    improved generalisation across different generators. The no-down
    architecture was used to avoid down sampling in the first ResNet50
    layer, preserving artifacts and noise residuals for better
    detection. The model combines global and local features using a
    patch selection module and attention-based fusion to enhance
    classification accuracy

2.  Spectrum-feature-based detection identifies unique patterns in
    AI-generated images by transforming them into spectrum space using
    fast Fourier transform. The Deep Image Fingerprint (DIF) method uses
    a U-Net high-pass filter to extract fingerprints and classifies
    images based on the correlation between these fingerprints and image
    residuals, achieving good performance even with limited training
    samples.

3.  Image-encoder-based detection uses the feature space of
    image-to-text encoders like CLIP, which are trained on large
    image--text datasets. The Universal Fake Image Detector (UFD)
    leverages CLIP's feature space with a linear classification layer to
    detect AI-generated images by comparing input features to a feature
    bank of real and fake images using cosine distance

- Label flipping attack was addressed by (Qiao et al., 2024) \[17\],
  fake images are detected by implementing unsupervised learning. Noise
  labels are generated based on image artifacts. Initially, images are
  clustered by extracting features from both the frequency and spatial
  domains. The frequency domain analysis uses the Discrete Cosine
  Transform (DCT) to reveal periodic artifacts that commonly appear in
  fake images. In the spatial domain, eye highlight consistency is
  evaluated by detecting facial landmarks with Dlib, extracting eye
  regions, and measuring the similarity of highlights between the two
  eyes using IoU scoring. These combined features enable the assignment
  of noisy labels without knowing the true image class. Next, the
  feature extractor is trained in two stages: pre-training with
  contrastive learning to make features of images with similar noisy
  labels close together (clustering)while separating different ones, and
  re-training using high-confidence samples to further refine the model.
  Finally, in the testing phase, features are extracted from images,
  clustered, and filtered using cosine similarity to effectively
  distinguish between real and fake images. This approach demonstrates
  how integrating frequency and spatial domain artifacts with
  contrastive learning can enable effective fake image detection without
  relying on labelled data.

# Research Methodologies

# This section discusses the overall research methodology adopted. It explains the research design, whether the approach is qualitative, quantitative, or mixed. The tools, technologies, and datasets used are described in detail, along with the frameworks, and models applied. The methods used for data collection are presented, followed by a justification for the chosen approaches. Finally, ethical considerations relevant to the study are discussed. {#this-section-discusses-the-overall-research-methodology-adopted.-it-explains-the-research-design-whether-the-approach-is-qualitative-quantitative-or-mixed.-the-tools-technologies-and-datasets-used-are-described-in-detail-along-with-the-frameworks-and-models-applied.-the-methods-used-for-data-collection-are-presented-followed-by-a-justification-for-the-chosen-approaches.-finally-ethical-considerations-relevant-to-the-study-are-discussed. .unnumbered}

**3.1 Type of Research Design**

**Research Philosophy**

Interpretivism is the philosophical worldview of this research because
it seeks to explain the uniqueness of morphed and deepfake detection in
digital media. Interpretivism focuses on the individual interpretation
of collected data and aims to explore the social, ethical, and
technological implications for the research problem (Junjie and Yingxin,
2022). By adopting this philosophy, the study aims to gain a deep
understanding of both the technical challenges and contextual issues in
detecting manipulated media, rather than relying solely on numerical
computation or automated detection scores.

**Research Approach**

A deductive approach is evident in this study, where theoretical
knowledge and data on digital forensics and image/video manipulation
detection are used to structure the investigation. With the bottom-up
approach, abstract concepts about morphing and deepfake generation
techniques are used to arrive at specific conclusions on detection
strategies (Hall, Savas-Hall, and Shaw, 2023). Its applicability ensures
that the research remains aligned with the objectives, enabling
efficient analysis of secondary data to validate existing theories and
suggest technological improvements for detection systems.

**Research Design**

**  
**An exploratory research design will be used to address the research
questions since detecting morphed and deepfake content is a rapidly
evolving and technically complex field. This design is particularly
useful when investigating new and emerging challenges such as
adversarial attacks on detection algorithms, ethical concerns, and
dataset creation for training models. The exploratory nature of the
research allows creativity in sampling, analyzing emerging issues, and
identifying potential opportunities for enhancing detection
technologies.

**Research Data Analysis**

The strength of the proposed methodology lies in its qualitative
research strategy, which involves the collection and analysis of
textual, descriptive, and interpretive data. Qualitative analysis is
effective in forming a theory-based overview of challenges and
strategies in morphed and deepfake detection (Lim, 2024). Industry
reports, technical evaluations, and case studies, combined with
literature reviews from scholarly publications, provide a strong
foundation for analysis. By focusing on qualitative patterns, the
research emphasizes relationships between detection methods,
technological limitations, and emerging manipulation techniques, which
can guide the development of more robust and reliable detection systems.

The proposed model also adopts a quantitative experimental design, as it
utilises numerical data such as image vectors for processing and
classification. This design enables the measurement of model performance
using statistical metrics including precision and F1-score. Quantitative
approaches are suitable for this study because they facilitate objective
evaluation and comparison of different machine learning models. The
research follows a deductive approach, beginning with the hypothesis
that Vision Transformers outperform traditional methods due to their
global attention mechanisms, and testing this hypothesis using
controlled experiments.

![A graph of different colored bars AI-generated content may be
incorrect.](media/image10.png){width="6.268059930008749in"
height="3.4666699475065617in"}

Fig: Comparing Siamese Network with other networks

The above graph is a sample image of the statistical analysis of a
Siamese network compared to other networks.

The data analysis in this study utilised a quantitative approach, where
the performance of the Siamese Network and Vision Transformer models was
evaluated using statistical metrics such as accuracy, precision, recall,
and F1-score. Confusion matrices were generated to analyse true and
false classifications in detail. Loss and accuracy graphs were plotted
to observe the models learning behaviour over training epochs. These
analyses were conducted using Python with libraries such as
Scikit-learn, Matplotlib, and NumPy to ensure objective and systematic
evaluation of model performance.

**Objective approach:**

Work is based on **facts, experiments, and quantitative evaluation**,
aiming to find, note, compare the about model performance.

**Tools and technologies used**

- **Python**: Main language

- **OpenCV**: Reading images (cv2.imread)

- **Pillow**: Opening images for hashing

- **imagehash**: Detecting duplicate images

- **HTML: Frontend**

- **CSS: Frontend**

- **JAVA SCRIPT: Frontend**

**Research Gap**

A greater difference is observed in terms of datasets, as the methods
used for feature extraction in morphed and deepfake detection are still
underdeveloped compared to other fields, or there are simply no large,
standardized datasets that can adequately represent real-world
manipulations. This creates another critical gap: the absence of
generally accepted criteria for assessing the authenticity of morphed
and deepfake content.

To fill these gaps, this research seeks to establish a survey on the
latest methods in manipulation detection, such as deep neural networks,
multimodal forensic approaches, and explainable AI models. It will also
assess the viability of developing niche datasets specifically for
deepfake and morphed image/video detection and recommend best practices
for dataset acquisition, benchmarking, and application. By addressing
these areas, the study aims to contribute significant findings that will
foster the advancement of more accurate, transparent, and unbiased
detection systems for digital media applications in security, forensics,
and everyday technology use.

**3.2 Data Collection**

This research is a secondary research approach primarily because the
information is obtained from secondary sources from different and
reliable sources such as peer-reviewed articles, technical reports, and
industry analyses. The dataset used in this study is taken from several
publicly available repositories, including the DeepFake Detection
Challenge (DFDC) dataset
(https://www.kaggle.com/c/deepfake-detection-challenge), the Celeb-DF
dataset (http://www.cs.albany.edu/\~lsw/celeb-deepfakeforensics.html),
the FaceSwap dataset (https://faceswap.dev/), the FaceForensics++ (FF++)
dataset (<https://github.com/ondyari/FaceForensics>), the
StyleGAN-generated faces (<https://github.com/NVlabs/stylegan2>), and
the Deepfake and Real Images dataset from Kaggle
(<https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images>).
A combined total of approximately 17,500 images was selected to ensure
coverage of different manipulation types while maintaining a manageable
dataset size. Only still images were included in the study; no video
data was processed. The images were further processed and the size of
data set has 4000 images at the end.

![A diagram of a set of blue
objects](media/image11.jpeg){width="6.795138888888889in"
height="1.4708333333333334in"}

**Fig 7**: Visual Representation of data collection from different data
sets

![](media/image12.png){width="5.779166666666667in"
height="2.5236111111111112in"}

Fig 8: Data collection and storage

**Data Preparation**

The image dataset was processed through several stages to ensure
quality, consistency, and suitability for the proposed hybrid
Siamese--Vision Transformer (ViT) model. All steps were implemented in
Python 3.10.12, using **OpenCV 4.8.0**, **Pillow 10.0.1**, **pandas
2.1.1**, **scikit-learn 1.3.1**, **TensorFlow 2.14.0**, and **imagehash
4.3.1**.

**1. Data Cleaning**

A custom function, check_image(), is implemented to open each file with
OpenCV's cv2.imread() function and corrupted or unreadable images were
identified. To address duplication, the perceptual hash (**pHash**)
algorithm was applied through the imagehash library (Yash Jakhar et al.,
2025) \[24\].

**2. Class Labelling**

Images were assigned to one of three categories --- *real*, *morphed*,
or *AI-generated* --- according to their source directories. This was
handled by assign_labels(), which mapped folder names to class IDs. The
filename--label pairs were stored in a CSV file created using
**pandas**, ensuring traceability and making it possible to verify the
dataset composition in later experiments.

**3. Image Resizing and Normalisation**

All images were standardised to a resolution of **224 × 224 pixels**
using cv2.resize() . This resolution because it is widely supported by
pretrained Vision Transformer architectures and provides a reasonable
balance between training time and detail retention. After resizing,
pixel values were converted from integers in the range \[0, 255\] to
floating-point values between 0 and 1, which helps stabilise neural
network training.

**4. Data Splitting**

The cleaned and labelled dataset was divided into training (60%),
validation (20%), and testing (15%) sets using scikit-learn's
train_test_split() function .To prevent data leakage, all images
originating from the same source were kept in the same split. This was
achieved through a mapping system implemented in
split_dataset_stratified().

**5. Data Augmentation**

Given the relatively small dataset size, augmentation was applied to
increase diversity and reduce overfitting. The augment_training_images()
function used TensorFlow's ImageDataGenerator to perform random
rotations (±20°), horizontal flips, brightness adjustments (±15%), and
cropping (90--100% of the original size). In addition, domain-specific
augmentations were introduced: random JPEG re-compression (quality range
70--100), Gaussian noise (σ = 0.01--0.05), and slight Gaussian blurring
(3×3 kernel). These were intended to mimic distortions and artefacts
that occur when images are uploaded to social media or compressed for
storage. (Shorten & Khoshgoftaar, 2019)\[25\]

**6. Format and Colour Space Conversion**

To maintain consistency, all images were converted to the RGB colour
space using Pillow's Image.convert(\"RGB\"). Images PNG were left in
thesame format during processing to avoid quality loss. The
convert_and_save_as_jpeg() function handled this step.

**7. Dataset Reduction and Justification**

The original cleaned dataset contained more images than could be
processed within the available time and computational limits. A subset
of **5,000 images** (1,250 per class) was selected for training and
evaluation. This reduction was carried out using random class-balanced
sampling while ensuring diversity across original sources. The primary
motivation was to conduct a **feasibility study** within the constraints
of the project schedule; however, it is recognised that training on a
larger dataset would likely improve performance, and this is identified
as a key area for future work.

**Summary**  
The final pre-processed dataset consisted of 5,000 RGB images, evenly
distributed across three classes, with standardised resolution, balanced
splits, and realistic augmentations. This prepared dataset provided a
consistent and fair input for the Siamese--ViT model, while the
preprocessing pipeline ensured reproducibility and minimised the risk of
bias or data leakage.

**3.3 Data Cleaning and Feature Extraction**

Images in csv format are imported to remove the background and just
extract the facial region using HarCascade classifier. Transformations
like rotations, zoom, width shift, height shift, shear, and horizontal
flip are used for the data augmentation.

![A diagram of a diagram AI-generated content may be
incorrect.](media/image13.jpeg){width="5.038319116360455in"
height="3.0991699475065615in"}

**Fig 9: Preparing the Data**

The data processing stage involved cleaning the dataset by removing
corrupted or mislabeled images, followed by resizing all images to a
standard input dimension suitable for the Siamese Network and Vision
Transformer models. Pixel values were normalized to a range of 0 to 1.
Data augmentation techniques such as rotation, flipping, and zooming
were applied to enhance dataset diversity and prevent overfitting.
images were converted into numerical tensors to be fed into the deep
learning models for training and evaluation.

![A diagram of data processing AI-generated content may be
incorrect.](media/image14.png){width="6.795138888888889in"
height="3.6944444444444446in"}

**Fig 10**: Demonstration of transfer learning

Dataset A is ImageNet, and Dataset B is the created dataset. The
proposed dataset (e.g., from DFDC, Face Forensics++, Celeb-DF, etc.)
contains real vs. fake faces. Fine-tuning on this dataset helps the
model specialize in detecting forgery-specific clues that were not
present in ImageNet.

From the table, we can see that the model with transfer learning and
finetune strategy performs best on DSTok datasets in this paper, as
expected. In addition, the model with only transfer learning has better
performance than the model trained from scratch. Through the results, we
show that the transfer learning and our finetune strategy are
effective.![A screenshot of a computer AI-generated content may be
incorrect.](media/image15.png){width="6.795138888888889in"
height="3.247916666666667in"}

Fig11: Represents a table proving Transfer learning is efficient than
training the model from scratch

**3.4 Justification for chosen methods**

**Why Siamese Network and ViT?**

Siamese network is implemented to compare real and fake images because
it outperforms models like RNNs, which are designed mainly for
sequential data such as videos and also require large datasets for
effective training. While CNNs are an option for image classification
tasks, implementing two parallel CNNs in a Siamese architecture is more
suitable here because it effectively compares the features of real and
fake images to detect morphs or forgeries.

In this proposed approach, if the Siamese network detects a morph, it
outputs the result directly. However, if it fails to detect the morph,
the image is passed to a second method: the Vision Transformer (ViT).
ViT analyses the frequency domain features of the image. Although there
are various techniques like texture analysis to detect morphs, studying
both spatial and frequency domains has shown higher accuracy in
distinguishing fake images.

![A table with text on it AI-generated content may be
incorrect.](media/image16.jpeg){width="5.267209098862642in"
height="1.8418591426071742in"}

While methods such as Non-local Neural Networks and Graph Neural
Networks can model global dependencies, Vision Transformers remain
preferable for this task due to their efficient global attention
mechanism, proven effectiveness in frequency domain artefact detection,
and ease of implementation with pretrained models. Although Fourier
Neural Operators offer direct frequency domain learning, they are yet to
be widely adopted in computer vision, whereas ViT integrates both
spatial and frequency features seamlessly for classification task.

To analyse artefacts, the Vision Transformer breaks the image into
standard-sized patches, typically 16×16 pixels. Each patch is then
converted into a vector embedding that includes positional information
about where the patch is located in the image. Additionally, ViT uses a
special classification token (\[CLS\]) that aggregates information from
all patches to understand the global structure and artefacts in the
image. This token then helps classify the image as real or fake.

The self-attention mechanism in ViT allows it to learn a global view of
the image, capturing long-range dependencies and large-scale differences
between real and fake images. In contrast, a CNN mainly captures local
features within its receptive field and may miss global inconsistencies,
limiting its ability to detect certain morphing artefacts.

**3.5 Ethical Considerations**

Smith and Miller (2022) discussed the possible concerns and issues
related to morphed and deepfake detection systems, especially in
sensitive categories such as digital forensics, media integrity, and
identity verification, including their ethical and social implications.
Misclassification of genuine content as fake, or failure to detect a
deepfake, could lead to wrongful accusations, reputational damage, or
denial of legitimate services, thereby depriving individuals of their
rights both socially and legally. Almeida, Shmarko and Lomas (2022)
critically opined that collecting and storing highly detailed biometric
or forensic markers, such as facial embeddings or behavioral patterns,
represents significant privacy risks given the sensitivity of the
information. These issues necessitate the establishment of ethical
policies and robust data protection standards to ensure that
developments in morphed and deepfake detection technologies remain
aligned with human rights and public trust. Considering the deployment
of detection systems in high-stakes contexts such as criminal justice,
digital evidence validation, and online media governance, there is a
lack of research with respect to ethical safeguards, social fairness,
and accountability.

The dataset used in this dissertation contains images of humans,
landscapes etc. The first concern is about the privacy. To ensure this,
all the images are gathered from public domain i.e., all the images are
sourced from datasets that were publicly available for research.

The second concern is bias in dataset. To overcome this bias, various
datasets were combined and a single dataset was formed.

The process transparency is maintained by documenting each step of
preprocessing and training in detail. I have also included the
limitations of the system so the results are not over-interpreted.

Finally, it's important to communicate responsibly. Even if the model
performs well on the test data, it doesn't mean it will be perfect in
the real world. I've made it clear that human judgement is still needed,
and this detection tool should not be relied on as the only line of
defence.

In summary, the methodology involved adopting a quantitative research
design under a positivist philosophy, collecting and processing image
datasets, analysing model performance using statistical metrics, and
addressing ethical, privacy, and methodological limitations with careful
consideration. These steps ensured the study was conducted
systematically and ethically to achieve its aim of detecting fake images
effectively.

**4.System Design**

**System Architecture**:

![A diagram of a company AI-generated content may be
incorrect.](media/image17.jpeg){width="8.031944444444445in"
height="3.99994750656168in"}

Fig 12: Proposed Hybrid model architecture

Source: Self-created (Canva)

The model operates in two phases:

Phase 1:Siamese Network

Conditions

- Need two input images(for D-MAD Differential Morphing Attack
  Detection)

1.  Original/real image

2.  Morphed image

- The input image must contain human face for comparison. Images of
  landscape and other images which do not contain human faces are
  directly sent to Vision Transformer in phase 2.

Phase 2: ViT (Vision Transformer)

Conditions

- Operates for single image input (S-MAD Self Morph Attack Detection)

- The image can be of a human, landscape or animal. (Anything)

![A diagram of a flowchart AI-generated content may be
incorrect.](media/image18.jpeg){width="5.409027777777778in"
height="2.986327646544182in"}

Fig13: Workflow of the model based on input image

Explanation:

If two images are given, the hybrid model operates and classifies image
as real or fake. If a single image is given only the ViT (Vision
Transformer) operates and classifies the image.

A Siamese Network consists of two CNN (Convolution Neural Network) in
parallel . Two ResNet-50 are implemented to form a Siamese Network.

![A diagram of different types of network AI-generated content may be
incorrect.](media/image19.jpeg){width="6.795138888888889in"
height="3.1659722222222224in"}

**Fig 14: Proposed Siamese Network Structure (Source: Self-Created
-Canva)**

![A person with glasses and a person with a face extract AI-generated
content may be
incorrect.](media/image20.jpeg){width="4.417929790026247in"
height="2.4565496500437445in"}

**Fig 15: Impact of face extraction using Hash function** A standard
frame size of 200\*200 is obtained

Using img_to_encoding() function images are resized to 160\*160
resolution. FaceNet takes this 160\*160 image as input, The output is a
matrix of shape (𝑚,128) that encodes each input face image into a
128-dimensional vector. Grayscaling was performed. Adaptive Unsharp mask
Guided filter (AUG) is utilized to pre-processed for reducing the noise
artifacts in images. Kavitha, P et al\[4\]. YOLOv4-tiny is used for
locating area of interest.

![A person with red eyes AI-generated content may be
incorrect.](media/image21.jpeg){width="4.039939851268591in"
height="2.5565791776027997in"}

**Fig 16 : Locating AOI (Area of Interest) using YOLOv4-tiny**

**4.1 Training and Testing**

Transfer learning methodology is used for training. Keras custom
construction model is used for training and testing. The captured facial
images pass through each layer of the proposed Siamese Network, undergo
Convolution, Batch Normalization,ReLU activation function (CBR) for
extracting target features. A pre-trained ResNet 50 is used in the
covolution layer. To fine-tune the pre-trained ResNet-50 model for
classification, a new fully connected head is added. It starts with a
7×7 Max Pooling layer to reduce the feature map size, followed by a
flatten layer that converts the output into a one-dimensional vector.
This vector is then passed to a dense layer with a ReLU activation
function for classification. (Baalaji, S.V. et al.) (2023) \[16\].
Feature extraction followed by similarity comparison, and advanced loss
functions for accurate classification. (Baalaji, S.V. et al.) (2023)

**[Similarity Comparison]{.underline}**

Similarity comparison is done using Euclidean distance formula. L is the
distance between feature vectors/embeddings obtained from extraction and
this is coverted into a value between 0 and 1.

![A collage of people with different colored lines AI-generated content
may be incorrect.](media/image22.jpeg){width="3.5in" height="1.65in"}

![A black and white math equation AI-generated content may be
incorrect.](media/image23.png){width="6.052929790026247in"
height="0.8126093613298337in"}

**Fig 17: Calculating Euclidean Distance**

**Source:** **Yuxin Niu and Zhongsheng Wang 2024 J. Phys.: Conf. Ser.
2872 012008**

> For a face image k , its encoding is 𝑓(k) , where 𝑓 is the function
> computed by the neural network.

- A is an \"Anchor\" image-a picture of a person.

- P is a \"Positive\" image-a picture of the same person as the Anchor
  image.

- N is a \"Negative\" image-a picture of the morphed image

- A fully connected layer coverts the embeddings into similarity score
  between 0 and 1.

> ![A diagram of a person\'s face AI-generated content may be
> incorrect.](media/image24.jpeg){width="5.935099518810149in"
> height="3.085249343832021in"}

Fig 18: Similarity Comparison Flow Chart

**[Contrastive Loss Function]{.underline}**

This function helps in minimizing the distance between the face images
of the person and maximizing the distance between morphed image.

![A math equation with black text AI-generated content may be
incorrect.](media/image25.png){width="6.268059930008749in"
height="2.125688976377953in"}

> **Fig :Loss function used**
>
> Source: Coolsheru (Github)\[17\]

**ViT Design:**

Vision Transformer (ViT) works differently from traditional CNNs.
Instead of scanning the image with filters, it breaks the image into
small patches and then treats these patches like words in a sentence and
dynamically frames filters. Using the self-attention mechanism, the
model learns which patches are important and how they relate to each
other.

The key part of ViT is called Multi-Head Self-Attention (MSA). This
allows the model to look at many parts of the image at the same time.
Each "head" focuses on different features, and then the results are
combined to create a stronger overall understanding of the image. This
helps the model capture both local details (like edges or textures) and
global patterns (like overall shapes or layouts). (Ali et al., 2023)

![A diagram of a vision transformer AI-generated content may be
incorrect.](media/image26.jpeg){width="6.704482720909886in"
height="4.650649606299212in"}

**Fig 19: Proposed ViT (Vision Transformer ) Architecture**

**Source:** Self -Created using Canva

**Reference:** (Lamichhane, 2025)\[13\]

The proposed ViT model works in three steps:

**[Step 1]{.underline}: Image Patching**

The initial input for ViT model is an image of size 224\*224. A class
named PatchEmbedding is created and the image passes through a function
for non-overlap patching. The patch size is initialized as 16\*16. The
entire image is divided into sequence of patches of size 16\*16.

A simple mathematical explanation is ,

Number of patches(m) = Image size/ Patch Size

Which is 224/16 =14 patches

Since ,we have 2 Dimensional 2-D images ,number of patches would double.

Finally,

Number of Patches (m)= Image size/ Patch Size\* 2

Which is 224/16 =14\* 2= 28 Patches.

![A computer screen shot of a program AI-generated content may be
incorrect.](media/image27.png){width="6.794035433070866in"
height="2.4772725284339456in"}

**Fig: Code snippet for image patching**

**Source:** **(Mohammedfahd)\[33\]**

The patches are then flattened into vectors. A vector of the shape (1,
p\* p\* c) where p\*p is the patch size and c represents the colour
channel (Red, Green, Blue). The feature vector of shape(m+1,d) is
generated using the formula in fig below. Every patch is embedded into a
vector. The position of each patch in the image is also stored . A
classification token \[CLS\] is added to the start of the sequence. This
token will act as the representation of the entire image. Each patch is
transformed into a fixed-dimensional vector using a learnable embedding
matrix E.(Wu et al., 2021)\[34\]

![A math equations and formulas AI-generated content may be
incorrect.](media/image28.png){width="4.201561679790026in"
height="2.0303029308836393in"}

Fig: Mathematical representation of feature vector (input to ViT\'s
Encoder)

Source: (Usmani et al., 2023)\[32\]

**[Step 2:]{.underline} Passing through** **Transformer Encoder**

A transformer usually consists of two blocks, encoder and decoder. The
encoder has self-attention block where the images patches are fed. The
self-attention block learns the relation between various pixels and
forwards to neural network which transforms the output into useful
representations. A decoder is used to reconstruct the image, but as per
the requirement of our model, the transformer architecture is modified.
The decoder block is eliminated and a classification head is
implemented.

![A diagram of a network AI-generated content may be
incorrect.](media/image29.png){width="5.770138888888889in"
height="2.852564523184602in"}

Fig 20: Architecture of Transformer Network

Source: (Usmani et al., 2023)\[32\]

![](media/image30.png){width="3.2587609361329832in"
height="5.486430446194226in"}

Fig 21: Proposed Transformer Encoder Architecture

Source: Self-created

The patch vectors and position embeddings are added using vector sum and
fed into encoder as input. Then these flattened patches undergo linear
projection for dimensionality reduction from 756 size to 512. The
dimensionality reduction increases the computation speed, reduces
processing load thus making the system more efficient. The reduced
vectors are fed to encoder. The multi-head self-attention in the encoder
contains three neural networks to generate query(Q), Key (K), value(V).

This involves the process of estimating how closely one element relates
to other elements. Three learnable weight matrices query, key and value
are used .

- Query (Q): Describes what each patch is attempting to identify or seek
  in relation to other patches.

- Key (K): Encodes the characteristics of each patch that can be
  compared with queries to determine how strongly they are related.

- Value (V): Holds the actual content of the patch, which is transferred
  forward in the attention mechanism if the patch is considered
  important.

> ![A blue and black background with many dots AI-generated content may
> be incorrect.](media/image31.png){width="6.795138888888889in"
> height="1.5208333333333333in"}

Query Network Key Network Value Network

Fig 22: Representation of Query,Key and value networks of a self
attention head in encoder

Each network takes the reduced flattened vectors as input and performs
dot product with a matrix containing random weights . The weights are
learned and improved during training. A matrix with random biases is
added at the end.

![A black background with white letters AI-generated content may be
incorrect.](media/image32.png){width="2.055554461942257in"
height="0.3472222222222222in"}

This is the formula for linear transformation used, where w is the
learned weight, b is the bias, x is the input value.

Once all the query, key and value matrices are obtained similarity
comparison is performed. Similarity score helps in estimating the amount
of attention to be given to a patch with respect to query. For example,
the query is searching for facial features in the entire input image,
the key values of all the patches are calculated and compared with query
values. If the key patch contains eyes, nose, mouth, face, skin, face
contour the similarity score is high indicating these features belong to
face. For any other irrelevant feature, the similarity score is low.

[**Self- attention Block**:]{.underline}

- 

- ![](media/image33.png){width="4.972222222222222in"
  height="0.20833333333333334in"}

- 

- 

- 

- 

![](media/image34.png){width="1.2291666666666667in" height="0.40625in"}
![](media/image35.png){width="1.6668996062992125in"
height="0.22919838145231847in"}
![](media/image36.png){width="1.2189206036745406in"
height="0.4375612423447069in"}

- 

> ![](media/image37.png){width="1.1458333333333333in"
> height="0.2847222222222222in"}
> ![](media/image37.png){width="1.0902734033245844in"
> height="0.3159722222222222in"}

- ![](media/image38.png){width="0.7813593613298337in"
  height="8.334536307961504e-2in"}

- ![A white background with black dots AI-generated content may be
  incorrect.](media/image39.png){width="0.9167946194225722in"
  height="0.5834142607174103in"}![A close up of a number AI-generated
  content may be
  incorrect.](media/image40.png){width="2.2086417322834646in"
  height="0.6146686351706037in"}

- 

The attention mechanism aims to model how different elements within a
sequence relate to one another. It does this by encoding each element
into a shared embedding space and then capturing both local interactions
and global dependencies, allowing the model to extract meaningful
features across the entire sequence. (Ali et al., 2023)\[31\]

**[Step 3:Passing through Multi-layer Perceptron(MLP)]{.underline}**

The output of transformer encoder is fed to MLP (Multi-Layer Perceptron)
as input. Each

perceptron is composed of fully connected layers, where each layer
applies a linear transformation followed by a nonlinear activation
function, ReLU (Rectified Linear Unit). These activation functions
introduce nonlinearity, allowing the model to capture complex feature
interactions that a purely linear transformation would fail to
represent. The MLP head learns all the spatial features from the
classification token (CLS) and labels the image as real or deepfake.

![A diagram of a function AI-generated content may be
incorrect.](media/image41.png){width="3.9027777777777777in"
height="1.5486111111111112in"} Fig 23: Architecture of a perceptron

**Working architecture of proposed Hybrid model**

The model takes two images as input , one real and another image that is
to be checked and classified as real or fake. The images undergo
pre-processing and the features are extracted and sent for similarity
comparison to Siamese Neural Network. Once the similarity comparison is
done based on the similarity score the image is classified as real or
morphed. The Siamese Network used in this model compares the facial
features and identifies any morphs in the face. If the Siamese Network
identifies as real image, the image is sent to another network for
second analysis. The image is passed through the ViT channel and
position encodings and patch embedding is performed followed by encoder
and decoder network for image reconstruction. During this process the
model learns the global artefacts and classifies images as real or
synthetic based on pixel relation. The accuracy is relatively high as
two step verification process is implemented.

![A screenshot of a computer AI-generated content may be
incorrect.](media/image42.jpeg){width="6.795138888888889in"
height="3.498611111111111in"}

**[Fig 24]{.underline}**: The interface page displayed when model runs

![A person with a beard and mustache AI-generated content may be
incorrect.](media/image43.jpeg){width="6.795138888888889in"
height="3.4770833333333333in"}

**[Fig 25]{.underline}**: Prediction of fake /synthetic image by hybrid
model

![A screenshot of a video game AI-generated content may be
incorrect.](media/image44.jpeg){width="6.795138888888889in"
height="3.375in"}

**Fig 26:** Prediction of real image by hybrid model

**[Various case studies based on inputs:]{.underline}**

**Case 1:**

**Input image 1:Image to be checked/Morphed image(Negative image)**

**Input Images 2: Real/authentic image(Anchor image)**

**Nature of morph: Face morph**

**Workflow:**

The anchor image and negative image are fed into the model. Based on the
similarity score the siamese network classifies the image as fake. In
the sample image given below the nose and lips are morphed , so when the
similarity score is measured it is 0.33 which is far less than 1 ,
0.33\<1. Therefore the image is classified as fake/morphed**.**

**Fig 27: Work flow of proposed model for morphed input**

![A screenshot of a diagram AI-generated content may be
incorrect.](media/image45.jpeg){width="6.086805555555555in"
height="4.083333333333333in"}

**Case 2:**

**Input image 1:Image to be checked/Morphed image(Negative image)**

**Input Images 2: Real/authentic image(Anchor image)**

**Nature of Morph: Background**

**Workflow:**

The anchor image and Negative image are given as input to the model. The
image passes through Siamese network for comparison, since the face is
not morphed, the Siamese network identifies negative image as real.
Since, the output is real the image undergoes second stage verification.
Now the ViT breakdown the image into 16\*16 sized image patches and
using self-attention module learns the pixel relation and identifies the
background of the image as edited, due to drastic change in pixel value
due to increased clarity of the background compared to the girl's face.
Therefore, the ViT classifies the image using global view as fake
/synthetic.

**FIG 28:Work flow of proposed model for deepfake input** ![A screenshot
of a computer AI-generated content may be
incorrect.](media/image46.jpeg){width="6.005244969378827in"
height="4.275in"}

> ![A collage of images of food and drinks AI-generated content may be
> incorrect.](media/image47.png){width="5.124375546806649in"
> height="2.8242694663167103in"}
>
> FIG 29: Taken from journal

**4.2 Potential Risks**

Potential risks in this study include privacy concerns related to
dataset usage, which were mitigated by ensuring that only publicly
available and licensed datasets were used. Security measures were
implemented to protect data storage and model files from unauthorized
access., The risk of model misuse was acknowledged, the purpose of this
research was strictly confined to detection rather than the generation
of fake images.

**4.3 Limitations of Methodology**

**Data Bias:**

**  
**Bias may result from secondary data sources in a way that is similar
to how limited scope and focus of available deepfake and morphed
datasets can introduce skewed outcomes (Shahbazi et al., 2023). To avoid
this, the study will rely on multiple sources, compiling data from
various credible and diverse repositories. Special instructions for
selecting sources of information will be followed to minimize reliance
on outdated or narrowly scoped datasets, ensuring a balanced foundation
for analysis.

**Ethical Concerns**:

The application of case studies concerning manipulated media raises
privacy and ethical concerns, as the use of morphed or deepfake examples
often involves sensitive identity information. To address this, the
research will ensure that all secondary data sources are ethical, legal,
and publicly available. In cases where sensitive information may appear,
appropriate measures such as anonymization or blurring of faces will be
applied, with strict conformity to regulations governing the handling of
such data.

**Time Management**:

The completion of research tasks may be affected by the evolving and
technically complex nature of deepfake detection studies (Hartmann and
Briskorn, 2022). To mitigate this, a thorough project timeline will be
implemented, supported by contingency plans, progress reporting, and
continuous checks against milestones. Consultations will also be
conducted periodically to evaluate progress and ensure deliverables are
completed within the set timeframe.

**Resource Constraints**:

Certain proprietary datasets, advanced detection benchmarks, and
industry reports from specialized firms may be inaccessible to the
researcher. To mitigate this risk, the study will rely on open-source
datasets, literature reviews, and academic resources. Collaboration with
institutions and research groups specializing in deepfake and digital
forensics detection will also be considered to enrich the study**.**

The methodology faced certain limitations, such as the lack of access to
extremely large and diverse datasets, which may limit the model's
ability to generalize to all types of fake images. Computational
constraints were addressed by leveraging cloud-based GPU resources for
training. Due to time limitations, the study focused on comparing
Siamese Networks and Vision Transformers, leaving further comparative
analysis with other models for future work.

The main challenge was to decide a method for real face recognition.
There are several methods like CNN,3D face recognition, thermal imaging.
Selecting trusted articles and filtering the results of various methods
consumed a lot of time.

Working with such large datasets required substantial computational
resources, leading to increased time and memory usage, and thus higher
costs. Based on my supervisor's advice, I decided to revise my approach
and focus instead on a Siamese model that uses two CNN's I had
successfully implemented. This allowed me to complete my research within
practical constraints while still delivering meaningful insights.

**5. Results and Evaluation**

This section presents a discussion of the results obtained from the
hybrid fake image detection model. First, the performances of the
Siamese Network and the Vision Transformer (ViT) were evaluated
individually. Next, the hybrid model, which combines both Siamese and
ViT architectures, was tested for its effectiveness in detecting fake
images. A comparison of the three models shows that the hybrid model
achieves higher accuracy than either of the individual models. The only
drawback of the hybrid approach is that it requires a larger dataset and
longer training time compared to the standalone models.

**5.1 Evaluation metrics**

The metrics used to evaluate the models are ,

- **Recall**

- **Precision**

- **Accuracy**

- **F1 Score**

**Recall**

Recall is calculated using True positives and false negatives. True
positives refer to the number of real images that the model classifies
as real without making mistakes. False Negatives refer to the number of
real images that the model classifies as fake.

Recall represents the proportion of actual positive cases that the model
successfully identifies. It is calculated using the formula:\[29\]

Recall=TP/TP+FN

where *TP* denotes True Positives and *FN* denotes False Negatives.

**Precision**

Precision is calculated using True positives and false positives. False
positives refer to the number of fake images that the model mistakenly
classified as real ones. Precision

refers to the proportion of predicted positive cases that are truly
positive.

It is calculated using the formula :\[29\]

Precision=TP/TP+FP

where *FP* represents False Positives, TP denotes True Positives.

**Accuracy**

Accuracy is calculated using True Positives, True Negatives, False
Positives, False Negatives. True Negatives are when fake images are
classified as fake correctly. It

provides an overall measure of correctness using the formula:\[29\]

Accuracy=TP+TN/TP+FP+FN+TN​

where *TN* indicates True Negatives.

**F1-score**

F1 Score is calculated using Precision and Recall

It is used as a balanced measure. It is the mean of Precision and
Recall.

F1=2 × (Precision \* Recall)/Precision+ Recall

This score is particularly useful when dealing with imbalanced datasets,
as it provides a single metric that accounts for both false positives
and false negatives.\[29\]

**5.2 Performance of Siamese Network**

Siamese Network is used in highly similar features classification.
Compared to other face recognition models like Mobilenetv2, Alex Net,
Rep VGG, the Siamese Network using ResNet50 is robust and achieves high
accuracy with low loss. The graph in fig depicts that Siamese Using
ResNet 50 has achieved highest accuracy i.e., it has classified more
than 90% of the real images accurately compared to other models. Also,
the contrastive loss function used produces very close vector embeddings
for similar images with low loss indicating if the image of a given
person is real or fake. The model uses the two input images converts
them into embeddings which are nothing but matrices containing the
converted image to binary numerical data. These embeddings are compared
and if the image is real the similarity score produced after is high and
if the similarity score is low the image is most likely a fake image or
morphed one. The maximum value for similarity score is 1 and the least
is 0. The proposed Siamese model has classified real images with
accuracy of 92%.

![A graph of different types of epichokes AI-generated content may be
incorrect.](media/image48.jpeg){width="5.511102362204724in"
height="2.4939009186351706in"}

Fig 30:The accuracy curves and loss curves of various face recognition
models

(Xiwen Zhang a b et al., 2024)\[25\]

The study by (Xiwen Zhang a b et al., 2024) \[25\], shows ResNet 50 is
the best model to be used as backbone used for feature extraction in our
Siamese Network. The results of Precision, Recall, F1-score and accuracy
of Siamese Network in our model prove the same. Precision of 0.72,
recall of 0.72, F1 score of 0.72, accuracy of 0.72 is achieved by using
Siamese neural network in real and fake image classification.

![A table with numbers and text AI-generated content may be
incorrect.](media/image49.png){width="6.519302274715661in"
height="1.0570516185476815in"}

Fig : Performance of Siamese Network

Source : *Hybrid deep learning model based on Gan and RESNET for
detecting fake faces \| IEEE Journals & Magazine \| IEEE Xplore*. \[29\]

The graph below shows the ROC curve of the ResNet-50 based Siamese model
for detecting real and fake images. The ROC curve compares the true
positive rate with the false positive rate at different thresholds,
which helps us see how well the model can separate the two classes. The
Area Under the Curve (AUC) is **0.69**, which means the model is better
than random guessing (AUC = 0.5) but still not very strong. A perfect
classifier would have an AUC close to 1.0, so this result suggests that
while the Siamese model has learned some useful features, it struggles
to handle more challenging cases, especially when the fake and real
images look very similar. Overall, the model shows potential but also
highlights the need for improvement, for example through fine-tuning,
using a bigger dataset, or combining it with other approaches like
Vision Transformers.

![A graph with a red line AI-generated content may be
incorrect.](media/image50.png){width="5.741779308836396in"
height="2.7623392388451444in"}

Fig 31 : False positive rate vs true positive rate for ResNet-50 network

**5.3 Performance of Vision Transformer(ViT)**

A ViT with transformer encoder outperforms the existing models in
detecting synthetic images which is evident from the table. A ViT stands
out because of its ability to learn global features of an image and its
self learning strategy.

![A table of numbers with black text AI-generated content may be
incorrect.](media/image51.png){width="5.9471948818897635in"
height="3.762820428696413in"}

Performance of various classifiers for fake image detection

Source:(Lamichhane, 2025) \[13\]

The Vision Transformer (ViT) model shows very strong performance in
detecting real and fake images, with an overall accuracy of **98.25%**.
Both real and fake classes have almost equal precision, recall, and
F1-scores (around **0.98**), which means the model is consistent and not
biased towards one class. The results suggest that ViT is highly
effective at capturing small differences between real and fake images,
making it reliable for deepfake detection

![A screenshot of a test results AI-generated content may be
incorrect.](media/image52.png){width="5.4359208223972in"
height="1.525640857392826in"}

Fig 32:Classification report of ViT on Deepfakes

Source: (Lamichhane, 2025) \[13\]

This confusion matrix demonstrates the performance of the Vision
Transformer (ViT) in detecting real and fake images. For the real
images, 37,831 were correctly classified as real (True Positives) while
249 were misclassified as fake (False Negatives). Similarly, out of the
fake images, 37,755 were correctly classified as fake (True Negatives)
whereas 326 were misclassified as real (False Positives). These results
indicate that the ViT model performs with high accuracy, with only a
very small number of misclassifications in both categories.

![A screenshot of a computer AI-generated content may be
incorrect.](media/image53.png){width="5.402777777777778in"
height="2.6041666666666665in"}

FIG 33: Confusion Matrix of Vision transformer

**5.4 Performance of proposed Hybrid model**

The performance of hybrid model is assumed to exceed the individual
performance of Siamese and ViT , as it covers both the local and global
artefacts. The proposed model combines the advantages of both
technologies. The synergistic effect between the generative capability
of GANs and the discriminative skills of RESNET improves the model's
ability to distinguish between authentic and fake faces. the Proposed
System showed a better trade-off between true positive and false
positive rates than both Siamese and ViT, despite having a little higher
false positive rate of 90.

ROC-AUC:

A ROC curve plotted using train dataset gave 90% accuracy indicating
that the model has learned a strong distinction between positive and
negative classes. This suggests that the classifier is able to rank true
positives above false positives in 90% of cases, showing the ability to
extract discriminative features from the training data (Fawcett, 2006).
This demonstrates that the model learned the patterns in fake images,
real images and that the feature extraction and classification stages
are consistent. An accuracy test using test dataset must be performed to
generalise the model's capability.

![A graph of a true positive rate AI-generated content may be
incorrect.](media/image54.jpeg){width="6.794187445319335in"
height="3.0576924759405073in"}

Fig 34: ROC curve of hybrid model on train dataset

ROC-AUC for test dataset:

On the test dataset, the model achieved a ROC-AUC score of 0.85,
indicating strong generalization. It is evident that the model can
classify real images from fake images with 85% accuracy, confirming its
practical reliability (Fawcett, 2006) but lower than the training
performance. From the results it can be concluded that there is a
moderate generalization gap, model has not severely overfitted to the
training data.

![A graph with a line and a red line AI-generated content may be
incorrect.](media/image55.jpeg){width="5.306365923009624in"
height="2.0319444444444446in"}

Fig 35: ROC -AUC of hybrid model on train dataset

Confusion Matrices:

Two confusion matrices, one representing the results of hybrid model on
train dataset and the other representing the results on test data set
are calculated.

The confusion matrix for the training dataset demonstrates that the
model correctly classified 1,580 real images and 1,568 fake samples, it
incorrectly classified 170 real images as fake and 168 fake images as
real. The model achieved about 90% accuracy and performed equally in
detecting both real and fake data. Since the number of false positives
and false negatives is nearly the same, the model is not biased.

The confusion matrix on test data set shows that 642 real images are
correctly labelled ,637 synthetic/tampered images are correctly labelled
as fake. 113 images are incorrectly predicted as fake and 108 images are
incorrectly predicted as real. The overall accuracy is 85% with very

low bias .

![A blue squares with white text AI-generated content may be
incorrect.](media/image56.jpeg){width="5.360320428696413in"
height="2.595833333333333in"}

FIG 36: Confusion matrix for test dataset

![A diagram of a test AI-generated content may be
incorrect.](media/image57.jpeg){width="5.719297900262467in"
height="3.3506944444444446in"}

FIG 37: Confusion matrix on Train dataset

s![A graph of a performance comparison AI-generated content may be
incorrect.](media/image58.jpeg){width="5.875in"
height="3.8194444444444446in"}

**FIG 38**:Bar graph to compare models performance on train and test
dataset

A bar graph plotted using the performance metics of train and test data
shows that the model performed better with trained data and there is 5%
difference in accuracy . This gap can be bridged by rigorously taining
the model with wide range of datasets containing various morphed images
and deepfakes with frequency and spatial artefacts .

**Comparision with other models**:

The table below represents performance of various models in fake image
classification in faceforences dataset. The model with CNN and CV it
achieved highest accuracy among three of them. The hybrid model of
Siamese and Vit can estimated Ly achieve higher accuracy because a
Siamese network is a combination of two CNN in parallel hence more
accuracy. The CNN based model alone can concentrate on eye region, CVit
can alone concentrate on face region, CNN and CVit together focus on
eyes and face but our proposed model can focus and learn from all
thefacial feature and can learn artefacts as well therby increasing the
accuracy.

- Wehrli et al. (2022) assessed that CNNs, for instance, have
  demonstrated orientation in recognizing specific facial structures
  within diverse groups but perform poorly when called upon to
  differentiate relatively similar faces. Furthermore, other techniques
  like hyperspectral imaging, which is able to pick data beyond the
  range of normal vision, hold a lot of potential, but they have not
  made their way into widespread usage mostly because of high cost and
  technicality in implementation.

![](media/image59.png){width="6.795138888888889in" height="0.8in"}

Source: (Soudy et al., 2024)

**Strengths of proposed method**:

The major strength of proposed method is it can identify both frequency
and spatial artefacts using ViT self -attention head.

A varied dataset containing different morphs like faceswap, expression
swap, features swap, deepfakes improved the quality of training thereby
making the model different kind of fakes.

The cascading structure of the pipeline acts as a twostep verification
in detecting synthetic images.If the Siamese Network fails, the ViT
detects the fake image . Therefore, the model is very accurate and
confidence is high.

**Limitations:**

The limitation of the project is it needs huge time for training since
both the Siamese and ViT are trained individually.

**Unanticipated findings:**

The performance of ViT is effected by the number of attention heads and
number of layers. It is usually assumed as number of sources like layers
or heads increase the accuracy of model increases but in case of ViT the
performance is maximum with 12 layers and 8 heads. Increasing or
decreasing them effect the performance which is evident in tables below.

![A table with numbers and text AI-generated content may be
incorrect.](media/image60.png){width="5.236842738407699in"
height="3.6368055555555556in"}

**Effect of number of layers on performance of ViT**

Source:Lamichhane, D. (2025)\[13\]

![A table with numbers and text AI-generated content may be
incorrect.](media/image61.png){width="5.073624234470691in"
height="3.4171434820647417in"}

**Effect of number of heads on performance of ViT**

Lamichhane, D. (2025)\[13\]

**Future scope**

Expand the model to detect fake videos .synthetic video generation
involves multiple pipelines and, unlike static images, introduces
additional complexities, such as temporal and inter-frame dependencies,
and stronger compression artifacts, all of which complicate the
detection process. (Battocchio et al., 2025) \[30\]

# References {#references .unnumbered}

1.  Morais, P., Domingues, I. and Bernardino, J. (2025) 'Deep learning
    techniques for detecting morphed face images: A literature review',
    IEEE Access, 13, pp. 105952--105981.
    doi:10.1109/access.2025.3578199.

2.  Venkatesh, S. et al. (2020) 'Detecting morphed face attacks using
    residual noise from deep multi-scale context aggregation network',
    2020 IEEE Winter Conference on Applications of Computer Vision
    (WACV), pp. 269--278. doi:10.1109/wacv45572.2020.9093488

3.  L. Qin, F. Peng, and M. Long, ''Face morphing attack detection and
    localization based on feature-wise supervision,'' IEEE Trans. Inf.
    Forensics Security, vol. 17, pp. 3649--3662, 2022

4.  Medvedev, I., Shadmand, F. and Gonçalves, N. (2023) 'Mordeephy: Face
    morphing detection via fused classification', Proceedings of the
    12th International Conference on Pattern Recognition Applications
    and Methods, pp. 193--204. doi:10.5220/0011606100003411

5.  Banerjee, S. and Ross, A. (2021) 'Conditional identity
    disentanglement for differential face Morph Detection', 2021 IEEE
    International Joint Conference on Biometrics (IJCB) \[Preprint\].
    doi:10.1109/ijcb52358.2021.9484355

6.  Blasingame, Z. and Liu, C. (2021) 'Leveraging adversarial learning
    for the detection of morphing attacks', 2021 IEEE International
    Joint Conference on Biometrics (IJCB), pp. 1--8.
    doi:10.1109/ijcb52358.2021.9484383

7.  Mousavi, Shokoufeh, Mostafa Charmi, and Hossein Hassanpoor.
    "Recognition of Identical Twins Based on the Most Distinctive Region
    of the Face: Human Criteria and Machine Processing Approaches."
    Multimedia tools and applications 80.10 (2021): 15765--15802. Web.
    <https://link.springer.com/article/10.1007/s11042-020-10360-3#Sec3>

8.  Chandana.S, Harini, and Senthil Kumar R. "A Deep Learning Model to
    Identify Twins and Look Alike Identification Using Convolutional
    Neural Network (CNN) and to Compare the Accuracy with SVM Approach."
    ECS transactions 107.1 (2022): 14109--14121. Web.
    <https://iopscience.iop.org/article/10.1149/10701.14109ecst/pdf>

9.  Hattab, A. and Behloul, A. (2023) Face-iris multimodal biometric
    recognition system based on deep learning - multimedia tools and
    applications, SpringerLink. Available at:
    <https://link.springer.com/article/10.1007/s11042-023-17337-y>

10. Alabdan, R. et al. (2024) 'Unmasking gan-generated faces with
    optimal deep learning and cognitive computing-based cutting-edge
    detection system', Cognitive Computation, 16(6), pp. 2982--2998.
    doi:10.1007/s12559-024-10318-9.

11. Baraheem, S.S. and Nguyen, T.V. (2023) 'Ai vs. AI: Can ai detect
    AI-generated images?', Journal of Imaging, 9(10), p. 199.
    doi:10.3390/jimaging9100199

12. Xu, Q., Jiang, X., Sun, T., Wang, H., Meng, L. and Yan, H. (2024).
    Detecting Artificial Intelligence-Generated images via deep trace
    representations and interactive feature fusion. Information Fusion,
    \[online\] 112, p.102578.
    doi:https://doi.org/10.1016/j.inffus.2024.102578

13. Lamichhane, D. (2025) 'Advanced detection of AI-generated images
    through Vision Transformers', *IEEE Access*, 13, pp. 3644--3652.
    doi:10.1109/access.2024.3522759.

14. Meng, Z. *et al.* (2024) 'Artifact feature purification for
    cross-domain detection of AI-generated images', *Computer Vision and
    Image Understanding*, 247, p. 104078.
    doi:10.1016/j.cviu.2024.104078.

15. Scherhag, U. *et al.* (2018) 'Detecting morphed face images using
    facial landmarks', *Lecture Notes in Computer Science*, pp.
    444--452. doi:10.1007/978-3-319-94211-7_48.

16. *Performance comparison and visualization of AI-generated-image
    detection methods \| IEEE Journals & Magazine \| IEEE Xplore*.
    Available at:
    https://ieeexplore.ieee.org/abstract/document/10508937/ (Accessed:
    23 July 2025).

17. Qiao, T. *et al.* (2024) 'Unsupervised generative fake image
    detector', *IEEE Transactions on Circuits and Systems for Video
    Technology*, 34(9), pp. 8442--8455. doi:10.1109/tcsvt.2024.3383833.

18. Wang, S. *et al.* (2023) 'A two-stage fake face image detection
    algorithm with expanded attention', *Multimedia Tools and
    Applications*, 83(18), pp. 55709--55730.
    doi:10.1007/s11042-023-17672-0.

19. Sabitha, R. *et al.* (2021a) 'Enhanced model for fake image
    detection (EMFID) using convolutional neural networks with histogram
    and wavelet based feature extractions', *Pattern Recognition
    Letters*, 152, pp. 195--201. doi:10.1016/j.patrec.2021.10.007.

20. Scherhag, U. *et al.* (2018a) 'Detecting morphed face images using
    facial landmarks', *Lecture Notes in Computer Science*, pp.
    444--452. doi:10.1007/978-3-319-94211-7_48.

21. Robertson, D.J., Kramer, R.S.S. & Burton, A.M. 2017, \"Fraudulent ID
    using face morphs: Experiments on human and automatic
    recognition\", *PLoS One, *vol. 12, no. 3.

22. Morais, P., Domingues, I. and Bernardino, J. (2025a) 'Deep learning
    techniques for detecting morphed face images: A literature review',
    *IEEE Access*, 13, pp. 105952--105981.
    doi:10.1109/access.2025.3578199.

23. Puzder, D. (2024) *The dangers of AI art and deepfakes*, *Office of
    Information Security*. Available at:
    https://informationsecurity.wustl.edu/the-dangers-of-ai-art-and-deepfakes/
    (Accessed: 23 June 2025).

24. Yash Jakhar *et al.* (2025) *Effective near-duplicate image
    detection using perceptual hashing and Deep Learning*, *Information
    Processing & Management*. Available at:
    https://www.sciencedirect.com/science/article/pii/S0306457325000287?utm_source=chatgpt.com
    (Accessed: 14 August 2025).

25. Shorten, C. and Khoshgoftaar, T.M. (2019) *A survey on image data
    augmentation for Deep Learning - Journal of Big Data*,
    *SpringerOpen*. Available at:
    https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0
    (Accessed: 15 August 2025).

26. Xiwen Zhang a b *et al.* (2024) *High-similarity sheep face
    recognition method based on a Siamese network with fewer training
    samples*, *Computers and Electronics in Agriculture*. Available at:
    https://www.sciencedirect.com/science/article/pii/S0168169924006860
    (Accessed: 18 August 2025).

27. *Proceedings of the first international conference on applied
    mathematics, statistics, and Computing (ICAMSAC 2023)* (no date)
    *Google Books*. Available at:
    https://books.google.co.uk/books?hl=en&lr=&id=9jIHEQAAQBAJ&oi=fnd&pg=PA51&dq=siamese%2Bnetwork%2Bresnet%2B50%2Bfake%2Bimage%2B&ots=nlGEEEpfv0&sig=iCzCdhZKuMlFHsp56jjXfBwkjds&redir_esc=y#v=onepage&q&f=false
    (Accessed: 20 August 2025).

28. *Siamese network features for Image matching \| ieee conference
    publication \| IEEE Xplore*. Available at:
    https://ieeexplore.ieee.org/abstract/document/7899663/ (Accessed: 20
    August 2025).

29. *Hybrid deep learning model based on Gan and RESNET for detecting
    fake faces \| IEEE Journals & Magazine \| IEEE Xplore*. Available
    at: https://ieeexplore.ieee.org/document/10562247/ (Accessed: 20
    August 2025).

30. Battocchio, J. *et al.* (2025) 'Advance fake video detection via
    Vision Transformers', *Proceedings of the ACM Workshop on
    Information Hiding and Multimedia Security*, pp. 1--11.
    doi:10.1145/3733102.3733129.

31. Ali, A.M. *et al.* (2023) 'Vision Transformers in image restoration:
    A survey', *Sensors*, 23(5), p. 2385. doi:10.3390/s23052385.

32. Usmani, S., Kumar, S. and Sadhya, D. (2023) *Efficient deepfake
    detection using shallow vision transformer - multimedia tools and
    applications*, *SpringerLink*. Available at:
    https://link.springer.com/article/10.1007/s11042-023-15910-z
    (Accessed: 23 August 2025). (Usmani et al., 2023)

33. Mohammedfahd (no date)
    *Pytorch-Collections/Building_Vision_Transformer_on_CIFAR_10_From_Scratch_Pytorch.ipynb
    at main · MOHAMMEDFAHD/Pytorch-Collections*, *GitHub*. Available at:
    https://github.com/MOHAMMEDFAHD/pytorch-collections/blob/main/Building_Vision_Transformer_on_CIFAR_10_From_Scratch_Pytorch.ipynb
    (Accessed: 23 August 2025).

34. Wu, J. *et al.* (2021) 'Vision transformer‐based recognition of
    diabetic retinopathy grade', *Medical Physics*, 48(12), pp.
    7850--7863. doi:10.1002/mp.15312.

35. Junjie, M. and Yingxin, M., 2022. The Discussions of Positivism and
    Interpretivism. *Online Submission*, *4*(1), pp.10-14.

36. Hall, J.R., Savas-Hall, S. and Shaw, E.H., 2023. A deductive
    approach to a systematic review of entrepreneurship literature.
    *Management Review Quarterly*, *73*(3), pp.987-1016

37. Lim, W.M., 2024. What is qualitative research? An overview and
    guidelines. *Australasian Marketing Journal*, p.14413582241264619.

38. Smith, M. and Miller, S., 2022. The ethical application of biometric
    facial recognition technology. *Ai & Society*, *37*(1), pp.167-175.

39. Wehrli, S., Hertweck, C., Amirian, M., Glüge, S. and Stadelmann,
    T., 2022. Bias, awareness, and ignorance in deep-learning-based face
    recognition. *AI and Ethics*, *2*(3), pp.509-522.

40. Almeida, D., Shmarko, K. and Lomas, E., 2022. The ethics of facial
    recognition technologies, surveillance, and accountability in an age
    of artificial intelligence: a comparative analysis of US, EU, and UK
    regulatory frameworks. *AI and Ethics*, *2*(3), pp.377-387.

41. Shahbazi, N., Lin, Y., Asudeh, A. and Jagadish, H.V., 2023.
    Representation bias in data: A survey on identification and
    resolution techniques. *ACM Computing Surveys*, *55*(13s), pp.1-39.

42. Hartmann, S. and Briskorn, D., 2022. An updated survey of variants
    and extensions of the resource-constrained project scheduling
    problem. *European Journal of operational research*, *297*(1),
    pp.1-14

43. Fawcett, T. (2006) 'An introduction to ROC analysis', *Pattern
    Recognition Letters*, 27(8), pp. 861--874.
    doi:10.1016/j.patrec.2005.10.010.

44. Soudy, A.H. *et al.* (2024) 'Deepfake detection using convolutional
    vision transformers and Convolutional Neural Networks', *Neural
    Computing and Applications*, 36(31), pp. 19759--19775.
    doi:10.1007/s00521-024-10181-7.

45. 

# Appendix A -- Ethical Approval {#appendix-a-ethical-approval .unnumbered}

| **Supervisor sign off**                    |     |
|--------------------------------------------|-----|
| Ethics form complete                       | ☒   |
| Ethical concerns acknowledged              | ☐   |
| Research tool(s) checked                   | ☐   |
| All relevant forms included (consent etc.) | ☐   |
| Is not high risk                           | ☒   |

# Student Project Approval Form {#student-project-approval-form .unnumbered}

> **[LD7083/ Computing and Digital Technologies Project: Student Project
> Approval]{.underline} [Form]{.underline}**
>
> You should use this document if you intend to use one of the existing
> module level approval ethics applications. Please complete this
> document and discuss your study with your supervisor before you
> collect any data. *Failure to complete this document and have all
> aspects signed off and approved by your supervisor risks a notable
> deduction in your grade and may risk a case of Academic misconduct.
> Please see the module Bb site for more details.*
>
> Please ensure that your project meets the conditions of the existing
> ethics application (available on Module Bb site). ***If it does not,
> then you will need to submit a full ethics application instead***.

<table>
<colgroup>
<col style="width: 49%" />
<col style="width: 50%" />
</colgroup>
<thead>
<tr class="header">
<th>Student Name:</th>
<th>Vasavi Swetha Sappa</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>Project Title:</td>
<td>Detecting fake images using CNN(Convolution Neural Network with
ResNet50 as backbone)</td>
</tr>
<tr class="even">
<td>Supervisor Name:</td>
<td>Zainab Ibrahim</td>
</tr>
<tr class="odd">
<td>Ethics application you are amending (check box):</td>
<td><ul>
<li><blockquote>
<p>Low-risk Lab-based research</p>
</blockquote></li>
</ul>
<blockquote>
<p><mark>↓✔</mark>Low Risk Secondary Data Science project</p>
</blockquote>
<ul>
<li><p>Medium Risk Secondary Data Science project from the private
domain required membership</p></li>
<li><blockquote>
<p>Questionnaire/ survey Study</p>
</blockquote></li>
<li><blockquote>
<p>Interview Study or other Usability Study</p>
</blockquote></li>
</ul></td>
</tr>
</tbody>
</table>

> **Introduction to the project:**
>
> **Methodology:** Please complete the table below, using the following
> info to guide you. Write this as a future tense method. Describe the
> **participants** that you will recruit, how many you are going to
> recruit, and indicate if you have any additional exclusion criteria.
> Include the **research design** (e.g. randomised/repeated
> measures/quantitative/qualitative/case study etc) and detail of your
> proposed **procedures** (i.e., how are you collecting the data?).
> Include information on all of the equipment you

<table>
<colgroup>
<col style="width: 49%" />
<col style="width: 50%" />
</colgroup>
<thead>
<tr class="header">
<th><blockquote>
<p>1. Is this a low-risk secondary data or lab- based study? If Yes
please go to questions 6 and 7.</p>
</blockquote></th>
<th><p><mark>↓✔</mark>YES</p>
<ul>
<li><blockquote>
<p>NO</p>
</blockquote></li>
</ul></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td><blockquote>
<p>2. Who are your participants and what is the inclusion criteria?</p>
</blockquote></td>
<td>NA</td>
</tr>
<tr class="even">
<td><blockquote>
<p>3. How many will you recruit and from where?</p>
</blockquote></td>
<td>NA</td>
</tr>
<tr class="odd">
<td><blockquote>
<p>4. Are there any exclusion criteria (reasons why people should not
participate)?</p>
</blockquote></td>
<td>NA</td>
</tr>
<tr class="even">
<td><blockquote>
<p>5. Research design:</p>
</blockquote></td>
<td>NA</td>
</tr>
<tr class="odd">
<td><blockquote>
<p>6. Procedures (describe what you will do to collect data, include all
equipment/methods you plan to use).</p>
</blockquote></td>
<td>I am planning to collect data from Github and Kaggle. I found two
datasets of fake images in Github that would exactly fit my
research.There are 400 images althogether. All the images are split into
two sets ,a test data and a training data and stored in the system.
Images in csv format are imported to remove the background and just
extract the facial region using HarCascade classifier. Transformations
like rotations, zoom, width shift, height shift, shear, and horizontal
flip are used for the data augmentation .</td>
</tr>
</tbody>
</table>

plan to use. If this is a low-risk study, outline how you will extract
data and list the criteria you will use to do this. Somebody should be
able to read this and replicate it. Describe all planned **data
analysis** for both quantitative (e.g. t-tests, ANOVA, correlation etc.)
and qualitative (content analysis, thematic analysis etc.) data. If
doing a low-risk study explain how you intend to analyze the data you
have collected. Use literature to justify your method

<table>
<colgroup>
<col style="width: 49%" />
<col style="width: 50%" />
</colgroup>
<thead>
<tr class="header">
<th>7. Data analysis methods:</th>
<th>Image Rescaling,Pixel Normalisation,Image Augmentation,Colour space
transformation,Contrast Normalisation,Tensor Transformation. Using
img_to_encoding() function images are resized to 160*160 resolution.
FaceNet takes this 160*160 image as input, The output is a matrix of
shape (𝑚,128) that encodes each input face image into a 128-dimensional
vector. Grayscaling was performed. Adaptive Unsharp mask Guided filter
(AUG) is utilized to pre-processed for reducing the noise artifacts in
images</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td><blockquote>
<p>8. Additional information:</p>
</blockquote></td>
<td></td>
</tr>
</tbody>
</table>

> **Health and Safety:** Relevant risk assessments are listed in the
> ethics application. If your project needs additional risk assessments,
> then you will need to submit a new ethics application. Please identify
> the elements of the listed risk assessment that are relevant for your
> study and the risk assessment(s) you are working with.
>
> Please check the relevant boxes\*:

- HL_RISK_173 Testing in an external environment

- HL_RISK_722 face to face interview

- HL_RISK_727 Group interview

<table>
<colgroup>
<col style="width: 25%" />
<col style="width: 40%" />
<col style="width: 33%" />
</colgroup>
<thead>
<tr class="header">
<th colspan="3"><p><strong>Areas of potential risk</strong></p>
<p><em>Please indicate how you will eliminate, or as a minimum
ameliorate, the following areas of potential risks throughout the
processes of research design, data generation, data analysis and
dissemination</em></p></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td><strong>Area of risk</strong></td>
<td><strong>Questions relating to this risk</strong></td>
<td><strong>How will you mitigate against this risk?</strong></td>
</tr>
</tbody>
</table>

| Avoiding harm to all involved in or potentially affected by the research | How will you ensure that your participants/ respondents come to no harm (psychological; emotional; physical). e.g. not subjecting them to questioning about sensitive issues without advance agreement? | The research doesn't involve any participants .The study involves no direct contact with individuals .Hence there is no risk involved.                                                      |
|--------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                                          | How will you ensure your own safety (beyond just physical) in undertaking the Enquiry?                                                                                                                  | The research involves creating a model with technical skill in python which is desk based.There is no potential harm of travelling or interacting with people.                              |
| Ensuring the anonymity of all participants/respo ndents                  | How will you ensure anonymity in collecting/generating data                                                                                                                                             | The data is pre collected by original source.There is no collection involved in this study.                                                                                                 |
|                                                                          | How will you ensure anonymity in reporting the data?                                                                                                                                                    | Data is pre-anonymized. The reported data is generic and does not include the names or any personal information of the participant.                                                         |
| Gaining informed consent from all participants / respondents             | How will you ensure participants consent in advance?provide consent forms.                                                                                                                              | The data is collected upon participants approval earlier by the original researchers.The original source allows researchers to use the data.                                                |
|                                                                          | (How) might participants/respondents be able to withdraw their data?                                                                                                                                    | Withdrawl of data is not possible because the participants cannot be identified /contacted .                                                                                                |
| Avoiding deception                                                       | How will you how you promote accuracy in recording, analysis, reporting of the data/findings?                                                                                                           | Accuracy is promoted by using secondary data sources, carefully recording and checking data, applying appropriate analysis methods, and reporting findings clearly with proper referencing. |
| Data storage and destruction                                             | How will you transport and store your data securely (e.g. password protected; cloud storage)                                                                                                            | Data used for the research is from public domain ,and is permitted to be used for research                                                                                                  |
|                                                                          | How will you destroy the data and when?                                                                                                                                                                 | Data is from public domain and will be available in future.                                                                                                                                 |
| Secondary data sets                                                      | *Is your data set(s) from a domain requires membership?*                                                                                                                                                | The dataset used in this study is publicly available on GitHub, an open-source platform that does not require any membership or special access permissions.                                 |
|                                                                          | *Does this data set can be used for educational or academic research purpose?*                                                                                                                          | Yes it can be used for academic research.                                                                                                                                                   |

> Please check this box after you have read and understood ethics and
> health and safety information.

[↓✔]{.mark} I confirm I have read the University's health and safety
policy and ethics policy. I have read and understood the requirement for
the mandatory completion of risk assessments and that my study does not
deviate from the module level approval ethics forms on Blackboard.

# Further information (add below, if applicable) {#further-information-add-below-if-applicable .unnumbered}

#  {#section-4 .unnumbered}

- Consent forms

- Participant information sheet

- Debrief form

- Recruitment materials

- Permission letters

- Data collection tools

<table>
<colgroup>
<col style="width: 74%" />
<col style="width: 25%" />
</colgroup>
<thead>
<tr class="header">
<th><p><strong>Student’s Name and sign</strong></p>
<p>Vasavi Swetha Sappa</p>
<p><strong>(Name)</strong></p></th>
<th><p><strong>Date</strong></p>
<p><strong>07/07/2025</strong></p></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td><p><strong>Supervisor’s name and sign</strong></p>
<p>Zainab</p>
<p><strong>Dr Zainab Ibrahim</strong></p></td>
<td><strong>Date: 11/7/25</strong></td>
</tr>
</tbody>
</table>

#  {#section-5 .unnumbered}

#   {#section-6 .unnumbered}

#  {#section-7 .unnumbered}

# Appendix B -- AI Usage Declaration {#appendix-b-ai-usage-declaration .unnumbered}

I declare that this submission contains contributions from AI software
and that it aligns with acceptable use as specified as part of the
assignment brief/ guidance and is consistent with good academic
practice.

> I acknowledge use of AI software to \[select as appropriate\]:
>
> □ Generate ideas or structure suggestions. \[insert AI tool(s) and
> links and/or how used\]

<table>
<colgroup>
<col style="width: 33%" />
<col style="width: 33%" />
<col style="width: 33%" />
</colgroup>
<thead>
<tr class="header">
<th><blockquote>
<p><strong>AI Tool</strong></p>
</blockquote></th>
<th><blockquote>
<p><strong>Links</strong></p>
</blockquote></th>
<th><blockquote>
<p><strong>How to use</strong></p>
</blockquote></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td></td>
<td></td>
<td></td>
</tr>
<tr class="even">
<td></td>
<td></td>
<td></td>
</tr>
<tr class="odd">
<td></td>
<td></td>
<td></td>
</tr>
</tbody>
</table>

> □ Write, rewrite, or paraphrase part of this submission. \[insert AI
> tool(s) and links\]

<table>
<colgroup>
<col style="width: 33%" />
<col style="width: 33%" />
<col style="width: 33%" />
</colgroup>
<thead>
<tr class="header">
<th><blockquote>
<p><strong>AI Tool</strong></p>
</blockquote></th>
<th><blockquote>
<p><strong>Links</strong></p>
</blockquote></th>
<th><blockquote>
<p><strong>How to use</strong></p>
</blockquote></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td></td>
<td></td>
<td></td>
</tr>
<tr class="even">
<td></td>
<td></td>
<td></td>
</tr>
<tr class="odd">
<td></td>
<td></td>
<td></td>
</tr>
</tbody>
</table>

> □ Generate any other aspect of the submission. \[insert AItool(s) and
> links\] / Include brief details\]

<table>
<colgroup>
<col style="width: 33%" />
<col style="width: 33%" />
<col style="width: 33%" />
</colgroup>
<thead>
<tr class="header">
<th><blockquote>
<p><strong>AI Tool</strong></p>
</blockquote></th>
<th><blockquote>
<p><strong>Links</strong></p>
</blockquote></th>
<th><blockquote>
<p><strong>How to use</strong></p>
</blockquote></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td></td>
<td></td>
<td></td>
</tr>
<tr class="even">
<td></td>
<td></td>
<td></td>
</tr>
<tr class="odd">
<td></td>
<td></td>
<td></td>
</tr>
</tbody>
</table>

> Signed: Date:

# Appendix C -- Meeting Logs {#appendix-c-meeting-logs .unnumbered}

#  {#section-8 .unnumbered}

**Record of Supervisory Meeting**

| Student Name: Vasavi Swetha Sappa | Programme: LD7083 |
|-----------------------------------|-------------------|
| Supervisor: Zainab Ibrahim        |                   |

<table>
<colgroup>
<col style="width: 37%" />
<col style="width: 62%" />
</colgroup>
<thead>
<tr class="header">
<th>Date &amp; starting/ ending time of Meeting:</th>
<th><p>06/06/2025</p>
<p>9:00 AM</p></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>Meeting Number:</td>
<td>01</td>
</tr>
<tr class="even">
<td>Mean of the meeting:</td>
<td>Online (Group Supervisory meeting)</td>
</tr>
<tr class="odd">
<td colspan="2"><p>Brief Summary of Discussion (200 words max):</p>
<p>A successful Master's postgraduate dissertation requires the
acquisition of relevant knowledge, a comprehensive understanding of
research methodologies, and the ethical application of these methods. It
is our responsibility to engage in regular meetings with their
supervisor, maintain a detailed meeting log, meet project milestones in
a timely manner, and promptly communicate any issues to the supervisor.
Supervisors provide constructive feedback throughout the research
process. The final dissertation should be between 10,000 and 12,000
words in length and must be structured to include the following
components: Abstract, Introduction, Literature Review, Research
Methodology, Data Analysis, Results, Evaluation, Conclusions and
Recommendations, as well as Appendices and a comprehensive list of
References.</p></td>
</tr>
<tr class="even">
<td colspan="2"><p>Agreed Actions:</p>
<p>Start drafting the abstract, introduction for the proposed project.
Check the feasibility of the project , should submit interim report in
July.</p>
<p>Student signature: Vasavi Swetha Sappa</p>
<p>Supervisor signature: Zainab Ibrahim</p></td>
</tr>
</tbody>
</table>

**Record of Supervisory Meeting**

| Student Name: Vasavi Swetha Sappa | Programme: LD7083 |
|-----------------------------------|-------------------|
| Supervisor: Zainab Ibrahim        |                   |

<table>
<colgroup>
<col style="width: 37%" />
<col style="width: 62%" />
</colgroup>
<thead>
<tr class="header">
<th>Date &amp; starting/ ending time of Meeting:</th>
<th><p>16/06/2025</p>
<p>9:00 AM</p></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>Meeting Number:</td>
<td>02</td>
</tr>
<tr class="even">
<td>Mean of the meeting:</td>
<td>Online (Group Supervisory meeting)</td>
</tr>
<tr class="odd">
<td colspan="2"><p>Brief Summary of Discussion (200 words max):</p>
<p>Finalize research</p>
<p>In this class professor discussed about Finalising reseach objectives
and possible Themes for Reviews. she explained that how can clearly
define' our reseach topics and main objectives</p>
<p>our reseach should be as clear as possible and quiet understandable
and it should connect thory and practice. It should be knowledable. it
will follow And SMART objectives.</p>
<p>5- specific, m-measurable, A - Achievable, R-Realistic T-Timely.</p>
<p>→ Revise your mind map, identity your scope and write Review
outline..</p></td>
</tr>
<tr class="even">
<td colspan="2"><p>Agreed Actions:</p>
<p>Start drafting the abstract, introduction for the proposed project.
Check the feasibility of the project , should submit interim report in
July.</p>
<p>Student signature: Vasavi Swetha Sappa</p>
<p>Supervisor signature: Zainab Ibrahim</p></td>
</tr>
</tbody>
</table>

**Record of Supervisory Meeting**

| Student Name: Vasavi Swetha Sappa | Programme: LD7083 |
|-----------------------------------|-------------------|
| Supervisor: Zainab Ibrahim        |                   |

<table>
<colgroup>
<col style="width: 37%" />
<col style="width: 62%" />
</colgroup>
<thead>
<tr class="header">
<th>Date &amp; starting/ ending time of Meeting:</th>
<th><p>23/06/2025</p>
<p>9:00 AM</p></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>Meeting Number:</td>
<td>03</td>
</tr>
<tr class="even">
<td>Mean of the meeting:</td>
<td>Online (Group Supervisory meeting)</td>
</tr>
<tr class="odd">
<td colspan="2"><p>Brief Summary of Discussion (200 words max):</p>
<p>ALS3:-Literature Review.</p>
<p>In this class professor explained about Literature Review and its
importance Literature Review is rervised for theoretical foundation and
critical Analysis.</p>
<p>It should address issues in your projects, gaps and similarities.. it
will also highlights any amendments and adjustments needs to be done
your project.</p>
<p>And in our project we always follow the 4C's concept. compare,
contrast, connect and concluse.</p>
<p>we need to identify if any gaps there and also need answer reseach
Questions-</p>
<p>Need to do peer-pair program and get the feedback of outcomes of
project and literature.</p>
<p>Ensure to Your review supports reasecarch design principles and main
goal offectives</p></td>
</tr>
<tr class="even">
<td colspan="2"><p>Agreed Actions:</p>
<p>Start drafting the literature review for the proposed project. Check
the feasibility of the project , should submit interim report in
July.</p>
<p>Student signature: Vasavi Swetha Sappa</p>
<p>Supervisor signature: Zainab Ibrahim</p></td>
</tr>
</tbody>
</table>

#  {#section-9 .unnumbered}

**Record of Supervisory Meeting**

| Student Name: Vasavi Swetha Sappa | Programme: LD7083 |
|-----------------------------------|-------------------|
| Supervisor: Zainab Ibrahim        |                   |

<table>
<colgroup>
<col style="width: 37%" />
<col style="width: 62%" />
</colgroup>
<thead>
<tr class="header">
<th>Date &amp; starting/ ending time of Meeting:</th>
<th><p>30/06/2025</p>
<p>9:00 AM</p></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>Meeting Number:</td>
<td>04</td>
</tr>
<tr class="even">
<td>Mean of the meeting:</td>
<td>Online (Group Supervisory meeting)</td>
</tr>
<tr class="odd">
<td colspan="2"><p>Brief Summary of Discussion (200 words max):</p>
<p>In this class Professor explained about how to write report.</p>
<ul>
<li><p>we need to submit the project approval form based on research
risk level e.g., low, medium.</p></li>
<li><p>Next she explained about Dissertation Report. In chapters 1, 2, 3
are treated as interim report.</p></li>
<li><p>we need submit first 3 chapters with approximately 5k words for
feedback. Total dissertation is 10k-20k words.</p></li>
<li><p>Introduction, Literature Review and Research methodology are the
chapters. we need explain precisely about cach and every anticrafts
about project.</p></li>
<li><p>After that we need schedule one to one meetings with supervisior
for regular updates and inputs. maintain a log for each
mecting.</p></li>
</ul></td>
</tr>
<tr class="even">
<td colspan="2"><p>Agreed Actions:</p>
<p>Submit ethics approval form and interim report and start designing th
emodel</p>
<p>Student signature: Vasavi Swetha Sappa</p>
<p>Supervisor signature: Zainab Ibrahim</p></td>
</tr>
</tbody>
</table>

**Record of Supervisory Meeting**

| Student Name: Vasavi Swetha Sappa | Programme: LD7083 |
|-----------------------------------|-------------------|
| Supervisor: Zainab Ibrahim        |                   |

<table>
<colgroup>
<col style="width: 37%" />
<col style="width: 62%" />
</colgroup>
<thead>
<tr class="header">
<th>Date &amp; starting/ ending time of Meeting:</th>
<th><p>07/07/2025</p>
<p>9:00 AM</p></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>Meeting Number:</td>
<td>05</td>
</tr>
<tr class="even">
<td>Mean of the meeting:</td>
<td>Online (Group Supervisory meeting)</td>
</tr>
<tr class="odd">
<td colspan="2"><p>Brief Summary of Discussion (200 words max):</p>
<p>The methodology and system design was explained. A methodology should
either be qualitative, quantative or mixed. The onion philosophy was
explained. Research design, type of methodology must be
documented.<br />
We should explain data collection and preparation methods.Should include
tools and technologies used.<br />
once finished, code implementation must be started .</p></td>
</tr>
<tr class="even">
<td colspan="2"><p>Agreed Actions:</p>
<p>Finish methodology section and proceed to design/implement the
system.</p>
<p>Student signature: Vasavi Swetha Sappa</p>
<p>Supervisor signature: Zainab Ibrahim</p></td>
</tr>
</tbody>
</table>

**Record of Supervisory Meeting**

| Student Name: Vasavi Swetha Sappa | Programme: LD7083 |
|-----------------------------------|-------------------|
| Supervisor: Zainab Ibrahim        |                   |

<table>
<colgroup>
<col style="width: 37%" />
<col style="width: 62%" />
</colgroup>
<thead>
<tr class="header">
<th>Date &amp; starting/ ending time of Meeting:</th>
<th><p>14/07/2025</p>
<p>9:00 AM</p></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>Meeting Number:</td>
<td>06</td>
</tr>
<tr class="even">
<td>Mean of the meeting:</td>
<td>Online (Group Supervisory meeting)</td>
</tr>
<tr class="odd">
<td colspan="2"><p>Brief Summary of Discussion (200 words max):</p>
<p>This sections was meant for clearing any doubts . The professor gave
us time to ask any queries and has explained ,cleared the doubts. I had
a problem in finding source for Vision transformer ,Prof.Zainab was
helpful and with her knowledge helped me.</p></td>
</tr>
<tr class="even">
<td colspan="2"><p>Agreed Actions:</p>
<p>Continue working on the design and start drafting the results.</p>
<p>Student signature: Vasavi Swetha Sappa</p>
<p>Supervisor signature: Zainab Ibrahim</p></td>
</tr>
</tbody>
</table>

**Record of Supervisory Meeting**

| Student Name: Vasavi Swetha Sappa | Programme: LD7083 |
|-----------------------------------|-------------------|
| Supervisor: Zainab Ibrahim        |                   |

<table>
<colgroup>
<col style="width: 37%" />
<col style="width: 62%" />
</colgroup>
<thead>
<tr class="header">
<th>Date &amp; starting/ ending time of Meeting:</th>
<th><p>4/08/2025</p>
<p>15:45 PM to 16;05 PM</p></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>Meeting Number:</td>
<td>07</td>
</tr>
<tr class="even">
<td>Mean of the meeting:</td>
<td>Online (One to one Supervisory meeting)</td>
</tr>
<tr class="odd">
<td colspan="2"><p>Brief Summary of Discussion (200 words max):<br />
I got feedback for my interim report . Few suggestion included:</p>
<ul>
<li><p>Change of dissertation title</p></li>
<li><p>Adding more references to literature review</p></li>
<li><p>Adding introduction to every section</p></li>
</ul></td>
</tr>
<tr class="even">
<td colspan="2"><p>Agreed Actions:</p>
<p>Implementing changes professor has suggested.</p>
<p>Student signature: Vasavi Swetha Sappa</p>
<p>Supervisor signature: Zainab Ibrahim</p></td>
</tr>
</tbody>
</table>

**Record of Supervisory Meeting**

| Student Name: Vasavi Swetha Sappa | Programme: LD7083 |
|-----------------------------------|-------------------|
| Supervisor: Zainab Ibrahim        |                   |

<table>
<colgroup>
<col style="width: 37%" />
<col style="width: 62%" />
</colgroup>
<thead>
<tr class="header">
<th>Date &amp; starting/ ending time of Meeting:</th>
<th><p>11/08/2025</p>
<p>15:20 PM to 15:40 PM</p></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>Meeting Number:</td>
<td>08</td>
</tr>
<tr class="even">
<td>Mean of the meeting:</td>
<td>Online (One to one Supervisory meeting)</td>
</tr>
<tr class="odd">
<td colspan="2"><p>Brief Summary of Discussion (200 words max):<br />
My doubts were clarified. I had doubt regarding :</p>
<ul>
<li><p>How to build a cascading pipeline of Siamese and ViT</p></li>
<li><p>How to test the model</p></li>
<li><p>How to present Viva</p></li>
</ul></td>
</tr>
<tr class="even">
<td colspan="2"><p>Agreed Actions:</p>
<p>Testing the model and comparing the results.</p>
<p>Student signature: Vasavi Swetha Sappa</p>
<p>Supervisor signature: Zainab Ibrahim</p></td>
</tr>
</tbody>
</table>

**Record of Supervisory Meeting**

| Student Name: Vasavi Swetha Sappa | Programme: LD7083 |
|-----------------------------------|-------------------|
| Supervisor: Zainab Ibrahim        |                   |

<table>
<colgroup>
<col style="width: 37%" />
<col style="width: 62%" />
</colgroup>
<thead>
<tr class="header">
<th>Date &amp; starting/ ending time of Meeting:</th>
<th><p>18/08/2025</p>
<p>15:20 PM to 15:40 PM</p></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>Meeting Number:</td>
<td>09</td>
</tr>
<tr class="even">
<td>Mean of the meeting:</td>
<td>Online (One to one Supervisory meeting)</td>
</tr>
<tr class="odd">
<td colspan="2"><p>Brief Summary of Discussion (200 words max):</p>
<p>An overview of my work was done. I explained the details of my
project and got few suggestions.</p></td>
</tr>
<tr class="even">
<td colspan="2"><p>Agreed Actions:</p>
<p>Format the file and follow dissertation template.</p>
<p>Student signature: Vasavi Swetha Sappa</p>
<p>Supervisor signature: Zainab Ibrahim</p></td>
</tr>
</tbody>
</table>

#  {#section-10 .unnumbered}
