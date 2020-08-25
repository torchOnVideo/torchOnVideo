# Miscellaneous Resources for Video on Pytorch


### PyVideoResearch
Has references to many datasets, dataloaders and models for video. Can be an excellent references
https://github.com/gsig/PyVideoResearch


### Torchvision.video
Has many dataloaders predefined for datasets, including kinetic
Docs at: ```https://pytorch.org/docs/stable/torchvision/index.html```
an old video describing the characteristics ``` https://www.youtube.com/watch?v=ECaQSXG_MBw ```


## Inspiration
There has been a great boom and consumption of videos in recent years. As expected, the focus of the deep learning community towards video-based applications has also increased manifold. While brainstorming for ideas we realised that there is a need for a unified library tackling the various subtasks of video-based deep learning which could be a great utility for beginners and experts alike.

## What it does
We provide an all-encompassing library that provides models, training and testing strategies, custom data set classes, metrics and many more utilities involving state-of-the-art papers of the various subtasks of video-based deep learning.

## How we built it
The subtasks/subdomains of this domain were completely new for most of us and hence we the first task was a highly intensive literature survey in order to understand the importance, feasibility, and challenges in relation to our idea.

Post this, weekly chat calls to explain and enhance our understanding of the papers as well as code testing and review sessions helped us to further our code in a collaborative manner. 
There was always a constant tussle between providing functionality as well as providing ease of use and flexibility. However, a lot of hard work and multiple code iterations help us reach the current stage of the library.

## Challenges we ran into
- Understanding, debugging sources, and implementing new components of the library were highly tricky. The intricate nature of the State-of-the-art papers made us read through multiple iterations of the paper before we began actually building or extending code.
- Planning the actual structure of the code to enhance both the functionality as well as the ease of use was challenging. We needed to ensure that the library is simple and lucid to use for a beginner as well as an expert.

## Accomplishments that we're proud of
- Created a simple and clean interface covering a good number of subtasks. It is highly modular and extendible. We hope this helps to make practice and research on deep learning on video easy and accessible to the Community.
- Thoroughly understood the details and intricacies with the state of art papers and also raised more interest in this domain.
- Provided Support for a good number of video based datasets

## What we learned
- A unified library for deep learning on Video is an urgent need
- Building for state of the art papers is challenging and hence needs more support and contribution for active development of this library.
- It is a highly resource intensive domain and thus also needs intensive research on how to reduce there overheads. This will definitely spearhed the domain exponentially.
- There are still many more subtasks/subdomains to tackle and we are fired up to take these challenges at the earliest.

## What's next for torchOnVideo

- Provide actual mini data sets for all essential datasets so that anyone can actually begin and understand their models by an even more hands-on approach - especially for those with limited network and storage resources.
- Work on implementing even more extendible video dataset classes and loaders and figure ways in which PyTorch's  video io library can also be enhanced at the same time. Also add the functionality of video based samplers.
- Provide support for more subtasks and state of the art papers to futher the aim of this library.
*We intended to also add a video classification task* however decided to leave it aside for the time being due to an already amazing MMAction library
- Build in-depth multi GPU supporting components which were understood over the implementation of the current papers
- Involve the community and seek contribution to building this into a really amazing and valuable library!


