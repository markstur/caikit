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
"""
Tests for conversion between python dict and protobuf Struct
"""

# Third Party
from google.protobuf import struct_pb2
import pytest

# Local
from caikit.core.data_model.json_dict import dict_to_struct, struct_to_dict


from tests.data_model_helpers import reset_global_protobuf_registry

@pytest.fixture(autouse=True)
def auto_reset_global_protobuf_registry():
    """Reset the global registry of generated protos"""
    with reset_global_protobuf_registry():
        yield


def test_dict_to_struct_to_dict():
    """Make sure dict_to_struct can handle all variants"""
    raw_dict = {
        "int_val": 1,
        "float_val": 0.42,
        "str_val": "asdf",
        "bool_val": False,
        "null_val": None,
        "list_val": [2, 3.14, "qwer", True, None, [1, 2, 3], {"nested": "val"}],
        "dict_val": {"yep": "works"},
    }

    # Make sure the dict round trips correctly
    struct = dict_to_struct(raw_dict)
    round_trip = struct_to_dict(struct)
    assert round_trip == raw_dict

    # Make sure the struct representation looks right
    assert set(struct.fields) == set(raw_dict)
    assert all(
        getattr(struct.fields[key], struct.fields[key].WhichOneof("kind")) == val
        for key, val in raw_dict.items()
        if not isinstance(val, (list, dict, type(None)))
    )
    assert struct.fields["null_val"].WhichOneof("kind") == "null_value"
    assert struct.fields["null_val"].null_value == struct_pb2.NullValue.NULL_VALUE

    # FIXME: assert isinstance(struct.fields["dict_val"].struct_value, struct_pb2.Struct)
    print(struct.fields["dict_val"].struct_value.__class__.__name__, " == ", struct_pb2.Struct.__name__)
    assert struct.fields["dict_val"].struct_value.__class__.__name__ == struct_pb2.Struct.__name__

    assert len(struct.fields["dict_val"].struct_value.fields) == len(
        raw_dict["dict_val"]
    )

    print(type(struct.fields["list_val"]))
    print("-----")
    print(type(struct.fields["list_val"].list_value))
    print(struct.fields["list_val"].list_value.__class__)
    print(struct.fields["list_val"].list_value.__class__.__qualname__)
    print(struct.fields["list_val"].list_value.__class__.__module__)
    print(struct.fields["list_val"].list_value.__class__.__name__)
    print(type(struct_pb2.ListValue()))
    print(struct_pb2.ListValue.__class__)
    print(struct_pb2.ListValue.__class__.__qualname__)
    print(struct_pb2.ListValue.__class__.__module__)
    print(struct_pb2.ListValue.__class__.__name__)


    # assert isinstance(struct.fields["list_val"].list_value, struct_pb2.ListValue)
    print(struct.fields["list_val"].list_value.__class__.__name__, " == ", struct_pb2.ListValue.__name__)
    assert struct.fields["list_val"].list_value.__class__.__name__ == struct_pb2.ListValue.__name__

    assert len(struct.fields["list_val"].list_value.values) == len(raw_dict["list_val"])

    # FIXME: Different modules for proto3
    print(struct.fields["dict_val"].struct_value.__class__.__module__, " != ", struct_pb2.Struct.__module__)
    assert struct.fields["dict_val"].struct_value.__class__.__module__ == struct_pb2.Struct.__module__
    assert isinstance(struct.fields["list_val"].list_value, struct_pb2.ListValue)


def test_dict_to_struct_invalid_value():
    """Make sure that a ValueError is raised if a bad type is encountered"""
    with pytest.raises(ValueError):
        dict_to_struct({"foo": 1, "bar": {"baz": b"asdf"}})
