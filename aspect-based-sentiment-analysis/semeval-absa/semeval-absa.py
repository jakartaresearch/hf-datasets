# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# TODO: Address all TODOs and remove all explanatory comments
"""SemEval 2015: Aspect-based Sentiment Analysis"""


import csv
import json
import os

import datasets

_DESCRIPTION = """\
This dataset is built as a playground for aspect-based sentiment analysis.
"""

_HOMEPAGE = "https://alt.qcri.org/semeval2015/"

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_TRAIN_LAPTOP_URL = "https://drive.google.com/uc?id=1Zvh4bZOZgSkIHrrA5WVvyPQO6-wWk4xQ"
_VAL_LAPTOP_URL = "https://drive.google.com/uc?id=14NgRdqcEHFfki0z49iMR8wqOEBnqdLH9"
_TRAIN_RESTAURANT_URL = "https://drive.google.com/uc?id=1fx1fWemdTYjonYSVfX-vcgU3KQa7C85V"
_VAL_RESTAURANT_URL = "https://drive.google.com/uc?id=1fHD0USeUgiLrnTo6zvRajk8whvsTVdAX"

DOMAINS = ['laptop', 'restaurant']

class ABSAConfig(datasets.BuilderConfig):
    """SemEval 2015 - ABSA Configs"""
    
    def __init__(self, domain: str, **kwargs):
        if domain not in DOMAINS:
            raise ValueError(f"Invalild domain: {domain}. Available domains: {DOMAINS}",)
        
        name = domain
        super(ABSAConfig, self).__init__(name=name, description=_DESCRIPTION, **kwargs)
        
        self.domain = domain
        
        self.url_train = _TRAIN_LAPTOP_URL if domain == 'laptop' else _TRAIN_RESTAURANT_URL
        self.url_val = _VAL_LAPTOP_URL if domain == 'laptop' else _VAL_RESTAURANT_URL


# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class ABSA(datasets.GeneratorBasedBuilder):
    """SemEval 2015: Aspect-based Sentiment Analysis."""

    _VERSION = datasets.Version("1.0.0")
    
    BUILDER_CONFIGS = [
        ABSAConfig(
            domain='laptop',
            version=_VERSION
        ),
        ABSAConfig(
            domain='restaurant',
            version=_VERSION
        )
    ]

    def _info(self):
        if self.config.domain == 'restaurant':
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "aspects": datasets.Sequence({
                        'term': datasets.Value("string"),
                        'polarity': datasets.Value("string"),
                        'from': datasets.Value("int16"),
                        'to': datasets.Value("int16"),
                    }),
                    "category": datasets.Sequence({
                        'category': datasets.Value("string"),
                        'polarity': datasets.Value("string")
                    })
                }
            )
        else:
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "aspects": datasets.Sequence({
                        'term': datasets.Value("string"),
                        'polarity': datasets.Value("string"),
                        'from': datasets.Value("int16"),
                        'to': datasets.Value("int16"),
                    })
                }
            )
#         features = datasets.Features(
#             {
#                 "id": datasets.Value("int16"),
#                 "text": datasets.Value("string"),
#                 "aspects": datasets.Sequence([{
#                     'term': datasets.Value("string"),
#                     'polarity': datasets.Value("string"),
#                     'from': datasets.Value("int8"),
#                     'to': datasets.Value("int8"),
#                 }]),
#                 "category": datasets.Sequence([{
#                     'category': datasets.Value("string"),
#                     'polarity': datasets.Value("string")
#                 }])
#             }
#         )
        
        
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE
        )

    def _split_generators(self, dl_manager):

        train_path = dl_manager.download(self.config.url_train)
        val_path = dl_manager.download(self.config.url_val)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": val_path})
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath):
        """Generate examples."""
        with open(filepath, 'r') as f:
            contents = json.load(f)
            for id_, row in enumerate(contents):
                yield id_, row