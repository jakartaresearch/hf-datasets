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
"""IndoQA: Indonesian Question Answering Dataset."""


import csv
import json
import os
import gdown

import datasets

_DESCRIPTION = """\
This dataset is built for question answering task.
"""

_HOMEPAGE = "https://github.com/jakartaresearch"

_TRAIN_URL = "https://drive.google.com/uc?id=1A-EqeT4GEYWv1cgkNnxMV8aTOvTxoMJQ"
_VAL_URL = "https://drive.google.com/uc?id=1oQENznKtAvt56JktD40GftHP7_7eVYYP"


class GooglePlayReview(datasets.GeneratorBasedBuilder):
    """IndoQA: Indonesian Question Answering Dataset."""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        
        features = datasets.Features(
            {
                "id": datasets.Value("string"),
                "context": datasets.Value("string"),
                "question": datasets.Value("string"),
                "answer": datasets.Value("string"),
                "category": datasets.Value("string"),
                "span_start": datasets.Value("int16"),
                "span_end": datasets.Value("int16")
            }
        )
        
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE
        )

    def _split_generators(self, dl_manager):

        train_path = dl_manager.download_and_extract(_TRAIN_URL)
        val_path = dl_manager.download_and_extract(_VAL_URL)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": val_path})
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath):
        """Generate examples."""
        with open(filepath, encoding="utf-8") as file:
            contents = json.load(file)
            for id_, row in enumerate(contents):
                yield id_, row