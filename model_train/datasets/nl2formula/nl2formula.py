# coding=utf-8
# Copyright 2021 The HuggingFace Datasets Authors and the current dataset script contributor.
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


import json
import os
from typing import List, Generator, Any, Dict, Tuple
import datasets


logger = datasets.logging.get_logger(__name__)

class Nl2formula(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="nl2vis",
            version=VERSION,
        ),
    ]

    def __init__(self, *args, writer_batch_size=None, **kwargs) -> None:
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features(
            {
                "Table_name":datasets.Value("string"),
                "Question": datasets.Value("string"),
                "Formula": datasets.Value("string"),
                "Level":datasets.Value("string"),
                "Table": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("string"))),
            }
        )
        return datasets.DatasetInfo(
            features=features,
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        #downloaded_filepath = dl_manager.download_and_extract(url_or_urls=_URL)
        downloaded_filepath = dl_manager.extract(path_or_paths="./t5-large/t5-base-new-1.zip")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_filepaths": [
                        os.path.join(downloaded_filepath, "t5-base-new-1/train.json"),
                    ]
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_filepaths": [os.path.join(downloaded_filepath, "t5-base-new-1/valid.json")],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_filepaths": [os.path.join(downloaded_filepath, "t5-base-new-1/test.json")],
                },
            ),
        ]

    def _generate_examples(
        self, data_filepaths: List[str]
    ) -> Generator[Tuple[int, Dict[str, Any]], None, None]:
        """This function returns the examples in the raw (text) form."""
        for data_filepath in data_filepaths:
            logger.info("generating examples from = %s", data_filepath)
            print(data_filepath)
            # we use the new formula
            with open(data_filepath, encoding="utf-8") as f:
                nl2formulas = json.load(f)
                id=0
                for idx, sample in enumerate(nl2formulas):
                    for formulas in sample["t5Formulas"]:
                        yield id,{
                            "Question":formulas["Question"],
                            "Formula":formulas["Formula2"],
                            "Table":sample["Table"],
                            "Table_name":sample["TableName"],
                            "Level":formulas["Level"]
                        }
                        id+=1
                    """
                    yield idx, {
                        "Question": sample["Question"],
                        "Formula":sample["Formula"],
                        "Formula2":sample["Formula2"],
                        "Table":sample["Table"]
                    }
                    """

