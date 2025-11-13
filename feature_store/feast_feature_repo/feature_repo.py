from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

# Offline source: generated training parquet (for demo)
source = FileSource(
    path="data/offline/train.parquet",
    timestamp_field=None,
)

user = Entity(name="row_id", join_keys=["row_id"])

features = FeatureView(
    name="tabular_features",
    entities=[user],
    ttl=timedelta(days=1),
    schema=[
        Field(name=f"f{i}", dtype=Float32) for i in range(20)
    ] + [Field(name="target", dtype=Int64)],
    online=True,
    source=source,
)
