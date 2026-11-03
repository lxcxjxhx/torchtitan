# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for DeepSeek-V4 pipeline-parallel stage splits."""

import unittest

import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.distributed.pipeline_parallel import (
    _split_module,
    get_module_fqns_per_model_part,
)
from torchtitan_recipes.tests.models import deepseek_v4 as config_registry


class TestDeepSeekV4PipelineSplit(unittest.TestCase):
    def test_two_stage_forward_matches_whole_model(self) -> None:
        torch.manual_seed(0)
        model_config = config_registry.deepseek_v4_debugmodel(seq_len=128).model
        model = model_config.build()
        model.init_weights(buffer_device=torch.device("cpu"))
        tokens = torch.randint(0, model_config.vocab_size, (128,))
        positions = torch.arange(64).repeat(2)  # two documents
        model_kwargs = {
            "positions": positions,
            "attention_metadata": model._get_attention_metadata(positions),
            "aux_loss_denominators": torch.tensor([128.0]),
        }
        # Debug model split: stage 0 = embeddings + hash-routed layers 0-1,
        # stage 1 = layers 2-3, hc_head, norm and lm_head.
        module_fqns_per_stage = get_module_fqns_per_model_part(
            model,
            first_stage_module_fqns=model.pipeline_first_stage_module_fqns,
            last_stage_module_fqns=model.pipeline_last_stage_module_fqns,
            parallelism_context=ParallelismContext(
                dp_replicate=1,
                dp_shard=1,
                cp=1,
                tp=1,
                pp=2,
                ep=1,
                world_size=2,
                enable_sequence_parallel=False,
            ),
            parallelism=ParallelismConfig(
                pipeline_parallel_degree=2, pipeline_parallel_schedule="1F1B"
            ),
            model_config=model_config,
        )
        first_stage, last_stage = (
            _split_module(model, module_fqns) for module_fqns in module_fqns_per_stage
        )

        hidden = first_stage(tokens, **model_kwargs)
        output = last_stage(hidden, **model_kwargs)

        self.assertEqual(hidden.shape, (128, model_config.hc_mult, model_config.dim))
        torch.testing.assert_close(
            output, model(tokens, **model_kwargs), rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
