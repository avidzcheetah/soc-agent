from typing import Optional, List, Any
from dataclasses import dataclass

@dataclass(kw_only=True)
class ToDictDataClass:
    def to_query_dict(self) -> dict[str, Any]:
        query: dict[str, str] = {}
        for key, value in vars(self).items():
            if value is None:
                continue
            if isinstance(value, list):
                query[key] = ",".join(
                    v.value if hasattr(v, "value") else str(v) for v in value
                )
            elif isinstance(value, bool):
                query[key] = str(value).lower()
            elif hasattr(value, "to_query_dict"):
                query[key] = value.to_query_dict()
            else:
                query[key] = str(value)
        return query

@dataclass
class CommonQueryParams(ToDictDataClass):
    pretty: bool = False
    wait_for_complete: bool = False


@dataclass
class PaginationQueryParams(ToDictDataClass):
    offset: Optional[int] = 0
    limit: Optional[int] = 500  # max 100000, recommended not to exceed 500
    sort: Optional[str] = (
        None  # string; use +/- for sorting order and dot notation for nested fields
    )
    search: Optional[str] = None  # string; prepend "-" for complementary search
    select: Optional[List[str]] = None
    q: Optional[str] = None  # Query string (e.g. 'status=active')