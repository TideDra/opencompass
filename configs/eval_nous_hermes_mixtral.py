from opencompass.models import VLLM
from mmengine.config import read_base
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
    from .summarizers.groups.mmlu import mmlu_summary_groups
    from .summarizers.groups.agieval import agieval_summary_groups
    from .summarizers.groups.bbh import bbh_summary_groups
    from .datasets.agieval.agieval_gen_64afd3 import agieval_datasets
    from .datasets.bbh.bbh_gen_5b92b0 import bbh_datasets
    from .datasets.truthfulqa.truthfulqa_gen import truthfulqa_datasets
datasets=[*bbh_datasets]

summarizer = dict(
    dataset_abbrs = [
        'bbh',
    ],
    summary_groups = bbh_summary_groups,
)

_meta_template = dict(
    round=[
        dict(role="HUMAN", begin='<|im_start|>user\n', end='<|im_end|>\n'),
        dict(role="BOT", begin="<|im_start|>assistant\n", end='<|im_end|>\n', generate=True),
    ],
)

models = [
    dict(
        type=VLLM,
        abbr='Nous-Hermes-2-Yi-34B',
        path='/mnt/gozhang/VL-RLHF/ckpts/tmp/Nous-Hermes-2-Yi-34B',
        model_kwargs=dict(tensor_parallel_size=4),
        meta_template=_meta_template,
        max_out_len=100,
        max_seq_len=2048,
        batch_size=32,
        generation_kwargs=dict(temperature=0),
        end_str='<|im_end|>',
        run_cfg=dict(num_gpus=4, num_procs=1),
    )
]