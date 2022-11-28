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
"""Google Play Review: An Indonesian App Sentiment Analysis."""


import csv
import json
import os

import datasets

_DESCRIPTION = """\
This dataset is built as a playground for beginner to make a use case for creating sentiment analysis model.
"""

_HOMEPAGE = "https://github.com/jakartaresearch"

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_TRAIN_URL = "https://media.githubusercontent.com/media/jakartaresearch/hf-datasets/main/cerpenmu/cerpen-corpus/small-cerpen.json"


# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class Indonews(datasets.GeneratorBasedBuilder):
    """Indonews: Multiclass News Categorization scrapped popular news portals in Indonesia.."""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        
        features = datasets.Features(
            {
                "title": datasets.Value("string"),
                "content": datasets.Value("string"),
                "url": datasets.Value("string"),
            }
        )
        
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE
        )

    def _split_generators(self, dl_manager):

        train_path = dl_manager.download_and_extract(_TRAIN_URL)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath):
        """Generate examples."""
        with open(filepath, encoding="utf-8") as file:
            data = json.load(file)
            for id_, row in enumerate(data):
                yield id_, row