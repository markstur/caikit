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
from typing import List, Optional

# Local
from caikit.core import DataObjectBase, dataobject
from typing import Dict, Union

# Type hints for JSON serializable dicts
RerankerJsonDictValue = Union[
    int,
    float,
    str,
    bool,
    type(None),
    List["RerankerJsonDictValue"],
    "RerankerJsonDict",
]
RerankerJsonDict = Dict[str, RerankerJsonDictValue]


@dataobject(package="caikit_data_model.caikit_nlp")
class RerankScore(DataObjectBase):
    """The score for one document (one query)"""

    document: Optional[RerankerJsonDict]
    index: int
    score: float
    text: Optional[str]


@dataobject(package="caikit_data_model.caikit_nlp")
class RerankQueryResult(DataObjectBase):
    """Result for one query in a rerank task.
    This is a list of n ReRankScore where n is based on top_n documents and each score indicates
    the relevance of that document for this query. Results are ordered most-relevant first.
    """

    query: Optional[str]
    scores: List[RerankScore]


@dataobject(package="caikit_data_model.caikit_nlp")
class RerankPredictions(DataObjectBase):
    """Result for a rerank tasks (supporting multiple queries).
    For multiple queries, each one has a RerankQueryResult (ranking the documents for that query).
    """

    results: List[RerankQueryResult]
