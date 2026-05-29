import re
from copy import deepcopy
from fractions import Fraction

from deepmerge import Merger
from dotmap import DotMap

from src.constants.common import FIELD_LABEL_NUMBER_REGEX
from src.defaults import CONFIG_DEFAULTS, TEMPLATE_DEFAULTS
from src.schemas.constants import FIELD_STRING_REGEX_GROUPS
from src.utils.file import load_json
from src.utils.validations import (
    validate_config_json,
    validate_evaluation_json,
    validate_template_json,
)

OVERRIDE_MERGER = Merger(
    # pass in a list of tuples,with the
    # strategies you are looking to apply
    # to each type.
    [
        # (list, ["prepend"]),
        (dict, ["merge"])
    ],
    # next, choose the fallback strategies,
    # applied to all other types:
    ["override"],
    # finally, choose the strategies in
    # the case where the types conflict:
    ["override"],
)


def get_concatenated_response(omr_response, template):
    """Concatenate multi-strip custom labels into final output values.

    Returns ``(concatenated_response, partial_read_detected)``.

    ``partial_read_detected`` is ``True`` when any composite field (e.g.
    ``CandidateNumber`` assembled from ``cand1..cand10``) ended up with SOME
    strips empty while OTHER strips produced a value. A blind ``"".join``
    silently drops the empty strips, producing a deceptively-short string
    (the production bug where prefill ``9010292074`` was read as ``"78"``
    because eight columns' ROIs landed in white space after a degraded
    3-marker warp). That is never a legitimate student-fill pattern for an
    integer-digit composite field, so the partial result is wrapped in
    ``MR(...)`` and the caller routes the sheet to MultiMarkedFiles for
    manual review instead of writing a silently-truncated value to Results.

    An ALL-empty composite (the student never filled the field) is left as
    the empty string — that is a legitimate "no response" pattern, not a
    partial-read failure.
    """
    concatenated_response = {}
    partial_read_detected = False
    strict_fields = getattr(template, "strict_composite_fields", set()) or set()
    for field_label, concatenate_keys in template.custom_labels.items():
        strip_values = [omr_response[k] for k in concatenate_keys]
        raw = "".join(strip_values)
        if field_label in strict_fields:
            non_empty_count = sum(1 for v in strip_values if v != "")
            if 0 < non_empty_count < len(strip_values):
                partial_read_detected = True
                concatenated_response[field_label] = f"MR({raw})"
                continue
        concatenated_response[field_label] = raw

    for field_label in template.non_custom_labels:
        concatenated_response[field_label] = omr_response[field_label]

    return concatenated_response, partial_read_detected


def open_config_with_defaults(config_path):
    user_tuning_config = load_json(config_path)
    user_tuning_config = OVERRIDE_MERGER.merge(
        deepcopy(CONFIG_DEFAULTS), user_tuning_config
    )
    validate_config_json(user_tuning_config, config_path)
    # https://github.com/drgrib/dotmap/issues/74
    return DotMap(user_tuning_config, _dynamic=False)


def open_template_with_defaults(template_path):
    user_template = load_json(template_path)
    user_template = OVERRIDE_MERGER.merge(deepcopy(TEMPLATE_DEFAULTS), user_template)
    validate_template_json(user_template, template_path)
    return user_template


def open_evaluation_with_validation(evaluation_path):
    user_evaluation_config = load_json(evaluation_path)
    validate_evaluation_json(user_evaluation_config, evaluation_path)
    return user_evaluation_config


def parse_fields(key, fields):
    parsed_fields = []
    fields_set = set()
    for field_string in fields:
        fields_array = parse_field_string(field_string)
        current_set = set(fields_array)
        if not fields_set.isdisjoint(current_set):
            raise Exception(
                f"Given field string '{field_string}' has overlapping field(s) with other fields in '{key}': {fields}"
            )
        fields_set.update(current_set)
        parsed_fields.extend(fields_array)
    return parsed_fields


def parse_field_string(field_string):
    if "." in field_string:
        field_prefix, start, end = re.findall(FIELD_STRING_REGEX_GROUPS, field_string)[
            0
        ]
        start, end = int(start), int(end)
        if start >= end:
            raise Exception(
                f"Invalid range in fields string: '{field_string}', start: {start} is not less than end: {end}"
            )
        return [
            f"{field_prefix}{field_number}" for field_number in range(start, end + 1)
        ]
    else:
        return [field_string]


def custom_sort_output_columns(field_label):
    # Audit fix CORE-14: previously this raised IndexError on field labels
    # that did not match FIELD_LABEL_NUMBER_REGEX (e.g. ``ROLL_NO``,
    # ``section-a``), aborting template construction with a confusing
    # traceback. Fall back to sorting the raw label when no match is found.
    matches = re.findall(FIELD_LABEL_NUMBER_REGEX, field_label)
    if not matches:
        return [field_label, 0]
    label_prefix, label_suffix = matches[0]
    return [label_prefix, int(label_suffix) if len(label_suffix) > 0 else 0]


def parse_float_or_fraction(result):
    if type(result) == str and "/" in result:
        result = float(Fraction(result))
    else:
        result = float(result)
    return result
