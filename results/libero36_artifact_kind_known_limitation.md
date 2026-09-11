# Known limitation: `artifact_kind` provenance field

The per-task HDF5 manifests for the LIBERO-36 pilot data may retain a historical `artifact_kind` value inherited from the Task 0 or Task 16 pilot templates. This field is provenance metadata only. It is not read by the training loader and is never used as a task label.

Training semantics are taken from the real `task_id` and the `language_instruction` stored in each task manifest/HDF5 and in the LeRobot `tasks.parquet` table. The semantic preflight confirms all 32 included tasks preserve these values and that no `artifact_kind` reference exists in the training conversion loader.

This limitation is documented for reproducibility; correcting the historical metadata label is not required for the frozen dataset or its training inputs.
