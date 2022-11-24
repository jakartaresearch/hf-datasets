"""Causal QA : """
import json
import csv
import yaml

import datasets

class CausalqaConfig(datasets.BuilderConfig):
    """BuilderConfig for causalqa."""

    def __init__(
        self,
        data_features,
        data_url,
        citation,
        **kwargs
    ):
        """BuilderConfig for GLUE.
        Args:
          data_features: `dict[string, string]`, map from the name of the feature
            dict for each text field to the name of the column in the tsv file
          data_url: `dict[string, string]`, url to download the zip file from
          citation: `string`, citation for the data set
          process_label: `Function[string, any]`, function  taking in the raw value
            of the label and processing it to the form required by the label feature
          **kwargs: keyword arguments forwarded to super.
        """
        super(CausalqaConfig, self).__init__(**kwargs)
        self.data_features = data_features
        self.data_url = data_url
        self.citation = citation

def OneBuild(data_info,feat_meta):
    main_name = [*data_info][0]
    submain_name = data_info[main_name].keys()
    all_config = []
    for k in submain_name:
        fm_temp = feat_meta[main_name][k]
        one_data_info = data_info[main_name][k]
        cqa_config = CausalqaConfig(
          name="{}.{}".format(main_name,k),
          description=one_data_info["description"],
          version=datasets.Version(one_data_info["version"], ""),
          data_features=fm_temp,
          data_url=one_data_info["url_data"],
          citation=one_data_info["citation"]
        )
        all_config.append(cqa_config)
    return all_config

_PATH_SOURCE = 'source/'
_PATH_METADATA = _PATH_SOURCE + 'features_metadata.yaml'
_FILE_URL = json.load(open(_PATH_SOURCE+'dataset_info.json'))
_CAUSALQA_DESCRIPTION = ''.join(open(_PATH_SOURCE+'dataset_description.txt').readlines())
_HOMEPAGE = _FILE_URL['homepage']

all_files = _FILE_URL['files']

class CausalQA(datasets.GeneratorBasedBuilder):
    """CausalQA: An QA causal type dataset."""
    with open(_PATH_METADATA, "r") as stream:
        try:
            fmeta = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    BUILDER_CONFIGS = []
    for f in all_files:
        BUILDER_CONFIGS += (OneBuild(f,fmeta))

    def _info(self):
        self.features = {feat: datasets.Value(self.config.data_features[feat]) 
                         for feat in self.config.data_features}
        
        return datasets.DatasetInfo(
            description=_CAUSALQA_DESCRIPTION,
            features=datasets.Features(self.features),
            homepage=_HOMEPAGE
        )

    def _split_generators(self, dl_manager):
        data_train = dl_manager.download(self.config.data_url['train'])
        data_val = dl_manager.download(self.config.data_url['val'])
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_train 
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
                feature_names = [*self.features]
                one_example_row = dict(zip(feature_names, existing_values))
                yield id_, one_example_row
