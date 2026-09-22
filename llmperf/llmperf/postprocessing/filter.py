from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llmperf.postprocessing.output import RequestOutput

class Filter:
    def __init__(self, category_ids: set[str] = None, category: str = None, aborted: bool = False):
        self.category_ids = category_ids
        self.category = category
        self.include_aborted = aborted

    def include(self, ro: "RequestOutput") -> bool:
        if not self.include_aborted and ro.aborted:
                return False
        if self.category_ids is not None and self.category is not None:
            return (ro.category is None and ro.id in self.category_ids) or ro.category == self.category
        if self.category_ids is not None:
            return ro.id in self.category_ids
        if self.category is not None:
            return ro.category == self.category
        return True