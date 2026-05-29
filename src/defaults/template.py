TEMPLATE_DEFAULTS = {
    "preProcessors": [],
    "emptyValue": "",
    "customLabels": {},
    "outputColumns": [],
    # Names of composite custom labels (those defined in customLabels) for
    # which a "partial" read - some strips empty while others are filled -
    # is geometrically suspicious rather than a legitimate short answer.
    # These fields are wrapped in MR(...) and routed to manual review
    # instead of silently producing a truncated value. Typical use: IDs
    # like CandidateNumber / RollNumber where every column must be filled.
    "strictCompositeFields": [],
}
