from jsonschema import Draft202012Validator

from scripts.build_professor_f1_gpt_dropoff_20260717 import _response_schema


def test_professor_response_schema_requires_every_named_axis_once():
    schema = _response_schema({"demo_case": ["A1", "A2"]})

    Draft202012Validator.check_schema(schema)
    axis_contract = schema["properties"]["cases"]["properties"]["demo_case"][
        "properties"
    ]["axis_assessments"]

    assert axis_contract["type"] == "object"
    assert axis_contract["additionalProperties"] is False
    assert axis_contract["required"] == ["A1", "A2"]
    assert set(axis_contract["properties"]) == {"A1", "A2"}
