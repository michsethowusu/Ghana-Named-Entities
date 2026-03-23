# 🇬🇭 Ghana Named Entities
A curated dataset of named entities extracted from Ghanaian news sources, 
compiled by the [Ghana NLP Community](https://huggingface.co/ghananlpcommunity).

## Dataset Description
Each record contains a named entity phrase found in Ghanaian news, along with 
its frequency count and a short human-readable description of what (or who) 
the entity refers to.

## Dataset Structure

| Column | Type | Description |
|--------|------|-------------|
| `phrase` | string | The named entity text as it appeared in the news |
| `count` | int | Number of times the entity was observed |
| `description` | string | A short description of the entity |

## Example

```python
from datasets import load_dataset
ds = load_dataset("ghananlpcommunity/ghana-named-entities")
print(ds["train"][0])
```

## Source
Entities were extracted from Ghanaian online news articles. 
The dataset is intended to support NLP research and applications 
focused on Ghanaian and West African contexts.

## License
Creative Commons Attribution 4.0 International (CC BY 4.0)

## Contributors
This dataset was created and curated by the following members of the Ghana NLP Community:

| Name | LinkedIn |
|------|----------|
| Kasuadana Sulemana Adams | [Profile](https://www.linkedin.com/in/gerhardt-datsomor/) |
| Mich-Seth Owusu | [Profile](https://www.linkedin.com/in/mich-seth-owusu/) |
| Jonathan Asiamah | [Profile](https://www.linkedin.com/in/jonathan-asiamah-4639a5147/) |
| Emmanuel Saah | [Profile](https://www.linkedin.com/in/emmanuel-saah/) |
| Gerhardt Datsomor | [Profile](https://www.linkedin.com/in/gerhardt-datsomor/) |

## Citation
If you use this dataset in your research or applications, please cite or acknowledge 
the Ghana NLP Community and its contributors:

```bibtex
@dataset{ghana_named_entities,
  author    = {Adams, Kasuadana Sulemana and Owusu, Mich-Seth and Asiamah, Jonathan and Saah, Emmanuel and Datsomor, Gerhardt},
  title     = {Ghana Named Entities},
  year      = {2024},
  publisher = {Ghana NLP Community},
  url       = {https://huggingface.co/datasets/ghananlpcommunity/ghana-named-entities}
}
```
