"""Nl2formula metrics."""

from typing import Optional, Union
#from seq2seq.metrics.spider.spider_test_suite import compute_test_suite_metric
#from seq2seq.metrics.spider.spider_exact_match import compute_exact_match_metric
import datasets


_DESCRIPTION = """
Nl2formula metrics.
"""



@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION)
class Nl2formula(datasets.Metric):
    def __init__(
        self,
        config_name: Optional[str] = None,
        keep_in_memory: bool = False,
        cache_dir: Optional[str] = None,
        num_process: int = 1,
        process_id: int = 0,
        seed: Optional[int] = None,
        experiment_id: Optional[str] = None,
        max_concurrent_cache_files: int = 10000,
        timeout: Union[int, float] = 100,
        **kwargs
    ):
        super().__init__(
            config_name=config_name,
            keep_in_memory=keep_in_memory,
            cache_dir=cache_dir,
            num_process=num_process,
            process_id=process_id,
            seed=seed,
            experiment_id=experiment_id,
            max_concurrent_cache_files=max_concurrent_cache_files,
            timeout=timeout,
            **kwargs
        )
        self.test_suite_db_dir: Optional[str] = kwargs.pop("test_suite_db_dir", None)

    def _info(self):
        if self.config_name not in [
            "exact_match",
            "test_suite",
            "both",
        ]:
            raise KeyError(
                "You should supply a configuration name selected in " '["exact_match", "test_suite", "both"]'
            )
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation="",
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                    "references":{
                        "context":datasets.Value("string"),
                        "label":datasets.Value("string"),
                        "level": datasets.Value("string")
                    }
                }
            ),
        )

    def _compute(self, predictions, references):
        if self.config_name == "exact_match" or self.config_name == "both":
            _exact_match = 0
            for prediction,reference in zip(predictions,references):
                if(prediction==reference['label']):
                    _exact_match+=1
            _exact_match=_exact_match/len(predictions)
            exact_match={"acc":_exact_match}

        else:
            exact_match = dict()

        if self.config_name == "test_suite" or self.config_name == "both":
            _test_suite=0
            easy,easy_true=0,0
            medium,medium_true=0,0
            hard,hard_true=0,0
            calculation,calculation_true=0,0
            for prediction,reference in zip(predictions,references):
                if(reference['level']=="easy"):
                    easy+=1
                elif(reference['level']=="medium"):
                    medium+=1
                elif(reference['level']=="hard"):
                    hard+=1
                else:
                    calculation+=1
                if(prediction==reference['label']):
                    _test_suite+=1
                    if (reference['level'] == "easy"):
                        easy_true += 1
                    elif (reference['level'] == "medium"):
                        medium_true += 1
                    elif (reference['level'] == "hard"):
                        hard_true += 1
                    else:
                        calculation_true += 1
            print("easy:", easy, "medium:", medium, "hard:",hard,"calculation:",calculation)
            _test_suite=_test_suite/len(predictions)
            easy_true=easy_true/easy
            medium_true=medium_true/medium
            hard_true=hard_true/hard
            if calculation!=0:
                calculation_true=calculation_true/calculation
            test_suite={"test_suite":_test_suite,
                        "easy":easy_true,
                        "medium":medium_true,
                        "hard":hard_true,
                        "calculation":calculation_true
                        }
        else:
            test_suite = dict()

        return {**exact_match, **test_suite}
