"""Causal QA : """

import json
import csv
import yaml

import datasets

_CAUSALQA_DESCRIPTION = """\
This dataset is built by xxx for causalqa
"""

_HOMEPAGE = "https://github.com/jakartaresearch"

## TODO: Add link to the official dataset URLs here
try:
  all_files = json.load(open('file_url.json'))['files']
except:
  all_files = json.load(open('causalqa/file_url.json'))['files']

def OneBuild(url_file,feat_meta):
    main_name = [*url_file][0]
    submain_name = url_file[main_name].keys()
    all_config = []
    for k in submain_name:
        fm_temp = feat_meta[main_name][k]
        cqa_config = CausalqaConfig(
          name="{}.{}".format(main_name,k),
          description="",
          version=datasets.Version("1.0.0", ""),
          text_features=fm_temp,
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
        **kwargs
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
        super(CausalqaConfig, self).__init__(**kwargs)
        self.text_features = text_features
        self.data_url = data_url
        self.citation = citation

# with open("features_metadata.yaml", "r") as stream:
#     try:
#         fmeta = yaml.safe_load(stream)
#     except yaml.YAMLError as exc:
#         print(exc)

# BUILDER_CONFIGS = []
# for f in all_files:
#     BUILDER_CONFIGS += (OneBuild(f, fmeta))
# print(BUILDER_CONFIGS[0].text_features)

class CausalQA(datasets.GeneratorBasedBuilder):
    """CausalQA: An QA causal type dataset."""
    with open("causalqa/features_metadata.yaml", "r") as stream:
        try:
            fmeta = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    BUILDER_CONFIGS = []
    for f in all_files:
        BUILDER_CONFIGS += (OneBuild(f,fmeta))

    def _info(self):
        
        features = {feat: datasets.Value(self.config.text_features[feat]) for feat in self.config.text_features} ## It assumes that everything is string
        
        return datasets.DatasetInfo(
            description=_CAUSALQA_DESCRIPTION,
            features=datasets.Features(features),
            homepage=_HOMEPAGE
        )

    def _split_generators(self, dl_manager):

        data_train = dl_manager.download(self.config.data_url['train'])
        data_val = dl_manager.download(self.config.data_url['val'])
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_train ## filepath or data_file?
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_val ## keys (as parameters) is used during generate example
                },
            )
        ]

    def _generate_examples(self, filepath):
        """Generate examples."""
        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            next(csv_reader)

            ## the yield depends on files features
            for id_, row in enumerate(csv_reader):
                existing_values = row
                feature_names = ['f'+str(i) for i in range(len(existing_values))]
                one_example_row =dict(zip(feature_names, existing_values))
                yield id_, one_example_row

