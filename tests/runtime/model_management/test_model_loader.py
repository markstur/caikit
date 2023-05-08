# Copyright The Caikit Authors
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
# Standard
import copy
from contextlib import contextmanager
import tempfile
import uuid

# Third Party
from unittest import mock

import grpc
import pytest

# Local
from caikit.config import get_config
from caikit.core import ModuleConfig
from caikit.core.blocks import base, block
from caikit.core.module_backends import BackendBase, backend_types
from caikit.core.module_backends.backend_types import register_backend_type
from caikit.runtime.model_management.batcher import Batcher
from caikit.runtime.model_management.model_loader import ModelLoader
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException
from sample_lib.blocks.sample_task import SampleBlock
from sample_lib.data_model import SampleInputType, SampleOutputType
from tests.conftest import temp_config
from tests.fixtures import Fixtures
from tests.core.helpers import MockBackend
import caikit.core.blocks

## Helpers #####################################################################


def _random_test_id():
    return "test-any-model-" + str(uuid.uuid4())


@contextmanager
def temp_model_loader():
    """Temporarily reset the ModelLoader singleton"""
    real_singleton = ModelLoader.get_instance()
    ModelLoader._ModelLoader__instance = None
    yield ModelLoader.get_instance()
    ModelLoader._ModelLoader__instance = real_singleton


@pytest.fixture
def model_loader():
    return ModelLoader.get_instance()


## Tests #######################################################################


def test_load_model_ok_response(model_loader):
    """Test that we can load up a valid model folder"""
    model_id = "happy_load_test"
    loaded_model = model_loader.load_model(
        model_id=model_id,
        local_model_path=Fixtures.get_good_model_path(),
        model_type=Fixtures.get_good_model_type(),
    )
    assert loaded_model.module() is not None
    assert isinstance(loaded_model.module(), base.BlockBase)
    assert model_id == loaded_model.id()
    assert Fixtures.get_good_model_type() == loaded_model.type()
    assert Fixtures.get_good_model_path() == loaded_model.path()

    # Models are not sized by the loader
    assert loaded_model.size() == 0


def test_load_model_archive(model_loader):
    """Test that we can load up a valid model archive"""
    model_id = "happy_load_test"
    loaded_model = model_loader.load_model(
        model_id=model_id,
        local_model_path=Fixtures.get_good_model_archive_path(),
        model_type=Fixtures.get_good_model_type(),
    )
    assert loaded_model.module() is not None
    assert isinstance(loaded_model.module(), base.BlockBase)


def test_load_model_error_not_found_response(model_loader):
    """Test load model's model does not exist error response"""
    model_id = _random_test_id()
    with pytest.raises(CaikitRuntimeException) as context:
        model_loader.load_model(
            model_id=model_id,
            local_model_path="test/this/does/not/exist.zip",
            model_type="categories_esa",
        )
    assert context.value.status_code == grpc.StatusCode.NOT_FOUND
    assert model_id in context.value.message


def test_load_invalid_model_error_response(model_loader):
    """Test load invalid model error response"""
    model_id = _random_test_id()
    with pytest.raises(CaikitRuntimeException) as context:
        model_loader.load_model(
            model_id=model_id,
            local_model_path=Fixtures.get_bad_model_archive_path(),
            model_type="not_real",
        )
    assert context.value.status_code == grpc.StatusCode.INTERNAL
    assert model_id in context.value.message


def test_it_can_load_more_than_one_model(model_loader):
    """Make sure we can load multiple models without side effects"""
    # TODO: change test to load multiple models

    model_id = "concurrent_load_test"
    model_1 = model_loader.load_model(
        model_id,
        Fixtures.get_good_model_path(),
        model_type=Fixtures.get_good_model_type(),
    )
    model_id = "concurrent_load_test_2"
    model_2 = model_loader.load_model(
        model_id,
        Fixtures.get_good_model_path(),
        model_type=Fixtures.get_good_model_type(),
    )

    assert model_1 is not None
    assert model_2 is not None
    # Different refs
    assert model_1 != model_2


def test_nonzip_extract_fails(model_loader):
    """Check that we raise an error if we throw in an archive that isn't really an archive"""
    model_id = "will_not_be_created"

    with pytest.raises(CaikitRuntimeException) as context:
        model_loader.load_model(
            model_id,
            Fixtures.get_invalid_model_archive_path(),
            Fixtures.get_good_model_type(),
        )
    # This ends up returning a FileNotFoundError from caikit core.
    # maybe not the best? But it does include an error message at least
    assert (
        context.value.status_code == grpc.StatusCode.NOT_FOUND
    ), "Non-zip file did not raise an error"
    assert Fixtures.get_invalid_model_archive_path() in context.value.message
    assert "config.yml" in context.value.message


def test_no_double_instantiation():
    """Make sure trying to re-instantiate this singleton raises"""
    with pytest.raises(Exception):
        ModelLoader()


def test_with_batching(model_loader):
    """Make sure that loading with batching configuration correctly wraps a
    Batcher around the model.
    """
    model = model_loader.load_model(
        "load_with_batch",
        Fixtures.get_good_model_path(),
        model_type="fake_batch_block",
    ).module()
    assert isinstance(model, Batcher)
    assert model._batch_size == get_config().runtime.batching.fake_batch_block.size

    # Make sure another model loads without batching
    model = model_loader.load_model(
        "load_without_batch",
        Fixtures.get_good_model_path(),
        model_type=Fixtures.get_good_model_type(),
    ).module()
    assert not isinstance(model, Batcher)


def test_with_batching_by_default(model_loader):
    """Make sure that a model type without specific batching enabled will
    load with a batcher if default is enabled
    """
    with temp_config({"runtime": {"batching": {"default": {"size": 10}}}}) as cfg:
        model = model_loader.load_model(
            "load_with_batch_default",
            Fixtures.get_good_model_path(),
            model_type=Fixtures.get_good_model_type(),
        ).module()
        assert isinstance(model, Batcher)
        assert model._batch_size == cfg.runtime.batching.default.size


def test_with_batching_collect_delay(model_loader):
    """Make sure that a non-zero collect_delay_s is read correctly"""
    model_type = Fixtures.get_good_model_type()
    with temp_config(
        {
            "runtime": {
                "batching": {
                    model_type: {
                        "size": 10,
                        "collect_delay_s": 0.01,
                    },
                }
            }
        }
    ) as cfg:
        model = model_loader.load_model(
            "load_with_batch_default",
            Fixtures.get_good_model_path(),
            model_type=model_type,
        ).module()
        assert isinstance(model, Batcher)
        assert model._batch_size == getattr(cfg.runtime.batching, model_type).size
        assert (
            model._batch_collect_delay_s
            == getattr(cfg.runtime.batching, model_type).collect_delay_s
        )


def test_load_distributed_impl():
    """Make sure that when configured, an alternate distributed
    implementation of a block can be loaded
    """

    reg_copy = copy.deepcopy(caikit.core.module.MODULE_REGISTRY)
    backend_registry_copy = copy.deepcopy(caikit.core.module.MODULE_BACKEND_REGISTRY)
    with mock.patch.object(caikit.core.module, "MODULE_REGISTRY", reg_copy) as registry_copy:
        with mock.patch.object(caikit.core.module, "MODULE_BACKEND_REGISTRY", backend_registry_copy) as block_registry_copy:
            @block(
                base_module=SampleBlock,
                backend_type=backend_types.MOCK,
                backend_config_override={"bar1": 1},
            )
            class DistributedGadget(caikit.core.blocks.base.BlockBase):
                """An alternate implementation of a Gadget"""

                SUPPORTED_LOAD_BACKENDS = [MockBackend.backend_type, backend_types.LOCAL]

                def __init__(self, bar):
                    self.bar = bar

                def run(self, sample_input: SampleInputType) -> SampleOutputType:
                    return SampleOutputType(greeting=f"hello distributed {sample_input.name}")

                @classmethod
                def load(cls, model_load_path) -> "DistributedGadget":
                    config = ModuleConfig.load(model_load_path)
                    return cls(bar=config.bar)

            with tempfile.TemporaryDirectory() as model_path:
                # Create and save the model directly with the local impl
                SampleBlock().save(model_path)

                model_type = "gadget"

                with temp_model_loader() as model_loader:
                    # Load the distributed version
                    model = model_loader.load_model(
                        "remote_gadget",
                        model_path,
                        model_type=model_type,
                    ).module()
                    assert isinstance(model, DistributedGadget)
