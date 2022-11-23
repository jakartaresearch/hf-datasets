"""Google Play Review: An Indonesian App Sentiment Analysis."""

import json
import os
import datasets

_CAUSALQA_DESCRIPTION = """\
This dataset is built by xxx for causalqa
"""

_HOMEPAGE = "https://github.com/jakartaresearch"

## TODO: Add link to the official dataset URLs here
all_files = json.load(open('file_url.json'))['files']

def OneBuild(url_file):
    main_name = [*url_file][0]
    submain_name = url_file[main_name].keys()
    all_config = []
    for k in submain_name:
        cqa_config = CausalqaConfig(
          name="{}.{}".format(main_name,k),
          description="",
          version='',
          text_features={"xx": "yy"},
          data_url=url_file[main_name][k],
          citation=""
        )
        all_config.append(cqa_config)
    return all_config

class CausalqaConfig(datasets.BuilderConfig):
    """BuilderConfig for causalqa."""

    def __init__(
        self,
        text_features,
        data_url,
        citation,
        **kwargs,
    ):
        """BuilderConfig for GLUE.
        Args:
          text_features: `dict[string, string]`, map from the name of the feature
            dict for each text field to the name of the column in the tsv file
          data_url: `dict[string, string]`, url to download the zip file from
          citation: `string`, citation for the data set
          process_label: `Function[string, any]`, function  taking in the raw value
            of the label and processing it to the form required by the label feature
          **kwargs: keyword arguments forwarded to super.
        """
        super(CausalqaConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)
        self.text_features = text_features
        self.data_url = data_url
        self.citation = citation

# _TRAIN_URL = "https://media.githubusercontent.com/media/jakartaresearch/hf-datasets/main/google-play-review/google-play-review/train.csv"
# _VAL_URL = "https://media.githubusercontent.com/media/jakartaresearch/hf-datasets/main/google-play-review/google-play-review/validation.csv"


# # TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class CausalQA(datasets.GeneratorBasedBuilder):
    """CausalQA: An QA causal type dataset."""
    BUILDER_CONFIGS = []
    for f in all_files:
        BUILDER_CONFIGS += (OneBuild(f))


    def _info(self):
        
        features = {text_feature: datasets.Value("string") for text_feature in self.config.text_features.keys()} ## It assumes that everything is string
        
        return datasets.DatasetInfo(
            description=_CAUSALQA_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE
        )

#     def _split_generators(self, dl_manager):

#         train_path = dl_manager.download_and_extract(_TRAIN_URL)
#         val_path = dl_manager.download_and_extract(_VAL_URL)
#         return [
#             datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}),
#             datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": val_path})
#         ]

#     # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
#     def _generate_examples(self, filepath):
#         """Generate examples."""
#         with open(filepath, encoding="utf-8") as csv_file:
#             csv_reader = csv.reader(csv_file, delimiter=",")
#             next(csv_reader)
#             for id_, row in enumerate(csv_reader):
#                 text, label, stars = row
#                 yield id_, {"text": text, "label": label, "stars": stars}