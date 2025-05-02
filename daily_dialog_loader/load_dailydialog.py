# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset"""

import os
from zipfile import ZipFile

import datasets


_CITATION = """\
@InProceedings{li2017dailydialog,
    author = {Li, Yanran and Su, Hui and Shen, Xiaoyu and Li, Wenjie and Cao, Ziqiang and Niu, Shuzi},
    title = {DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset},
    booktitle = {Proceedings of The 8th International Joint Conference on Natural Language Processing (IJCNLP 2017)},
    year = {2017}
}
"""

_DESCRIPTION = """\
We develop a high-quality multi-turn dialog dataset, DailyDialog, which is intriguing in several aspects.
The language is human-written and less noisy. The dialogues in the dataset reflect our daily communication way
and cover various topics about our daily life. We also manually label the developed dataset with communication
intention and emotion information. Then, we evaluate existing approaches on DailyDialog dataset and hope it
benefit the research field of dialog systems.
"""

_URL = "http://yanran.li/files/ijcnlp_dailydialog.zip"

act_label = {
    "0": "__dummy__",  # Added to be compatible out-of-the-box with datasets.ClassLabel
    "1": "inform",
    "2": "question",
    "3": "directive",
    "4": "commissive",
}

emotion_label = {
    "0": "no emotion",
    "1": "anger",
    "2": "disgust",
    "3": "fear",
    "4": "happiness",
    "5": "sadness",
    "6": "surprise",
}


class DailyDialog(datasets.GeneratorBasedBuilder):
    """DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset"""

    VERSION = datasets.Version("1.0.0")

    __EOU__ = "__eou__"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "dialog": datasets.Sequence(datasets.Value("string")),
                    "act": datasets.Sequence(datasets.ClassLabel(names=list(act_label.values()))),
                    "emotion": datasets.Sequence(datasets.ClassLabel(names=list(emotion_label.values()))),
                }
            ),
            supervised_keys=None,
            homepage="http://yanran.li/dailydialog",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        dl_dir = dl_manager.download_and_extract(_URL)
        data_dir = os.path.join(dl_dir, "ijcnlp_dailydialog")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "dialog_path": os.path.join(data_dir, "dialogues_train.txt"),
                    "act_path": os.path.join(data_dir, "dialogues_act_train.txt"),
                    "emotion_path": os.path.join(data_dir, "dialogues_emotion_train.txt"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "dialog_path": os.path.join(data_dir, "dialogues_validation.txt"),
                    "act_path": os.path.join(data_dir, "dialogues_act_validation.txt"),
                    "emotion_path": os.path.join(data_dir, "dialogues_emotion_validation.txt"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "dialog_path": os.path.join(data_dir, "dialogues_test.txt"),
                    "act_path": os.path.join(data_dir, "dialogues_act_test.txt"),
                    "emotion_path": os.path.join(data_dir, "dialogues_emotion_test.txt"),
                },
            ),
        ]

    def _generate_examples(self, dialog_path, act_path, emotion_path):
        """Yields examples."""
        with open(dialog_path, encoding="utf-8") as dialog_f, open(act_path, encoding="utf-8") as act_f, open(emotion_path, encoding="utf-8") as emotion_f:
            for idx, (dialog_line, act_line, emotion_line) in enumerate(zip(dialog_f, act_f, emotion_f)):
                dialog = dialog_line.strip().split(self.__EOU__)[:-1]
                act = act_line.strip().split(" ")[:-1]
                emotion = emotion_line.strip().split(" ")[:-1]

                assert len(dialog) == len(act) == len(emotion), f"Mismatch at line {idx}"

                yield idx, {
                    "dialog": dialog,
                    "act": [act_label[x] for x in act],
                    "emotion": [emotion_label[x] for x in emotion],
                }
