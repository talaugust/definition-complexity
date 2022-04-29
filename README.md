## Generating Scientific Definitions with Controllable Complexity


By Tal August, Katharina Reinecke, and Noah A. Smith

Abstract: Unfamiliar terminology and complex language can present barriers to understanding science. Natural language processing stands to help address these issues by automatically defining unfamiliar terms. We introduce a new task and dataset for defining scientific terms and controlling the complexity of gen- erated definitions as a way of adapting to a specific reader’s background knowledge. We test four definition generation methods for this new task, finding that a sequence-to-sequence approach is most successful. We then explore the version of the task in which definitions are generated at a target complexity level. We in- troduce a novel reranking approach and find in human evaluations that it offers superior fluency while also controlling complexity, compared to several controllable generation base- lines.

## Data

We use two sources for scientific definitions: 

* [MedQuAD](https://github.com/abachaa/MedQuAD)
* [Wikipedia Science Glossaries](https://en.wikipedia.org/wiki/Category:Glossaries_of_science)

For both sources, all terms and definitions are formatted as "What is (are) X?" with the answer being the definition of X.

TBD: `data/` includes files for the wikipedia entries. For the MedQuAD entries, please download the dataset from the [MedQuAD repository](https://github.com/abachaa/MedQuAD). Once the files are downloaded, you can use the code in `data/' to convert into the dataset used by our models. 


## Models
`models/` has training scripts for our models. All training and finetuning was done on a NVIDIA Titan X 12GB GPU. Details on hyperparameter settings are in the paper, though the scripts include the best performing settings we used. You can download our best-performing model (BART finetuned with the support documents) from the [HuggingFace Hub](https://huggingface.co/talaugust/bart-sci-definition). 

## Code
TBD: `generation/` includes notebooks for generating and evaluating definitions at different levels of complexity. 

## Citation

If you use our work, please cite our paper

```
@inproceedings{august-2022-definition-complexity,
    title={Generating Scientific Definitions with Controllable Complexity},
    author={Tal August, Katharina Reinecke, and Noah A. Smith},
    booktitle={ACL},
    year={2022}
}
```
