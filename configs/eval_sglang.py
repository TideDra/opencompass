from mmengine.config import read_base
from opencompass.models import SglangAPI
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    # 直接从预设数据集配置中读取需要的数据集配置
    from .datasets.piqa.piqa_ppl import piqa_datasets
    from .datasets.siqa.siqa_gen import siqa_datasets
    from .datasets.agieval.agieval_gen_64afd3 import agieval_datasets
    from .datasets.hellaswag.hellaswag_gen_6faab5 import hellaswag_datasets
    from .datasets.SuperGLUE_BoolQ.SuperGLUE_BoolQ_gen_883d50 import BoolQ_datasets
    from .datasets.mmlu.mmlu_gen_4d595a import mmlu_datasets

datasets=[*hellaswag_datasets,*mmlu_datasets]
api_meta_template = dict(
    round=[
            dict(role='HUMAN', api_role='HUMAN'),
            dict(role='BOT', api_role='BOT', generate=True),
    ],
)

models = [
    dict(abbr='llava-1.6-34b',
        type=SglangAPI,
        meta_template=api_meta_template,
        query_per_second=1,
        max_out_len=2048, max_seq_len=4096, batch_size=36,
        url="http://127.0.0.1:30000"),
]

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=8,
        task=dict(type=OpenICLInferTask)),
)
