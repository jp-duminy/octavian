"""

The circulatory system of Octavian, managing how user-configured stages are run and determining what raw data can be safely dropped between analysis stages.

"""

# all default libraries
from dataclasses import dataclass
from yaml import safe_load
from pathlib import Path
from itertools import product

@dataclass(frozen=True, slots=True)
class PipelineStage:
    """
    Object containing metadata about an Octavian analysis stage.
    """
    name:                    str
    requires:                frozenset[str]
    applies_to:              frozenset[str]
    needs_particle_columns:  frozenset[str]

@dataclass(frozen=True, slots=True)
class Internals:
    """
    Internal pipeline management and naming dictionaries for data writing.
    """
    stages: dict[str, PipelineStage]
    baryonic_ptypes: frozenset[str]
    group_keys: dict[str, str]  
    plist_to_ptype: dict[str, str] # eventually deprecate    
    group_ptype_lists: dict[str, list[str]]  

def load_internals(internals_filepath: Path, user_config: dict) -> dict[str, PipelineStage]:
    """
    Loads stage definitions from internals.yaml, validates output columns,
    and returns the stage registry.
    """
    with open(internals_filepath, "r") as f:
        internals = safe_load(f)

    output_columns = internals["output_columns"]
    stages: dict[str, PipelineStage] = {}

    for stage_name, stage_config in internals["stages"].items():
        stages[stage_name] = PipelineStage(
            name=stage_name,
            requires=frozenset(stage_config.get("requires", [])),
            applies_to=frozenset(stage_config["applies_to"]),
            needs_particle_columns=frozenset(stage_config.get("needs_particle_columns", [])),
        )

    all_stage_outputs: set[str] = set()

    for stage_name, stage_config in internals["stages"].items():

        for sub_block in stage_config.get("outputs", []):
            templates = sub_block["columns"]

            if "over" in sub_block:
                resolved_over = resolve_over(sub_block["over"], user_config)

                for key, values in resolved_over.items():
                    assert len(values) > 0, (
                        f"Stage {stage_name!r}: 'over' key {key!r} resolved to empty list. "
                        f"Please check the from_config: reference."
                    )

                expanded = expand_column_templates(templates, resolved_over)
            else:
                expanded = list(templates)

            for col_name in expanded:
                assert col_name in output_columns, (
                    f"Stage {stage_name!r} says it outputs {col_name!r}."
                    f"However, not found in output_columns."
                )
                assert col_name not in all_stage_outputs, (
                    f"Output {col_name!r} is being computed by multiple stages."
                )
                all_stage_outputs.add(col_name)

    # validate all output columns are claimed by a stage
    unclaimed = set(output_columns.keys()) - all_stage_outputs
    assert not unclaimed, f"output_columns not claimed by any stage: {unclaimed}"

    return Internals(
        stages=stages,
        baryonic_ptypes=frozenset(internals["baryonic_ptypes"]),
        group_keys=internals["groupIDs"],
        plist_to_ptype={v: k for k, v in internals["ptype_lists"].items()},
        group_ptype_lists=internals["group_ptype_lists"],
    )


def resolve_over(over: dict[str, list | str], user_config: dict) -> dict[str, list[str]]:
    """
    Expands the "over:" field in internals.yaml.
    """
    resolved: dict[str, list[str]] = {}

    for key, val in over.items():

        if isinstance(val, str) and val.startswith("from_config:"):

            config_key = val.split(":", maxsplit=1)[1]
            config_value = user_config[config_key]

            if isinstance(config_value, dict):
                config_value = list(config_value.keys()) # needed for the radial quantiles dict in config

            resolved[key] = [str(v) for v in config_value]

        else:
            resolved[key] = [str(v) for v in val]
            
    return resolved

def expand_column_templates(outputs: list[str],over: dict[str, list[str]]) -> list[str]:
    """
    Expands an output column template in internals.yaml if they have ptype or quantity-specific fields (e.g. mass_star_30kpc), returning a list.
    """
    expanded: list[str] = []

    for template in outputs:

        matched_keys = [key for key in over if f"{{{key}}}" in template] # triple bracket is needed to parse the .yaml {}

        if not matched_keys:
            expanded.append(template)
            continue

        value_lists = [over[key] for key in matched_keys]

        for combo in product(*value_lists):
            result = template

            for key, val in zip(matched_keys, combo):
                result = result.replace(f"{{{key}}}", str(val))

            expanded.append(result)

    return expanded

def get_releasable_columns(current_index: int, ordered_stages: list[PipelineStage]) -> frozenset[str]:
    """
    Returns a (frozen) set of which ParticleStore columns (the expensive ones) are no longer needed by any remaining stage and thus can be safely discarded.
    """
    current_needs = ordered_stages[current_index].needs_particle_columns
    future_needs: set[str] = set()

    for stage in ordered_stages[current_index + 1:]:
        future_needs |= stage.needs_particle_columns

    return current_needs - future_needs

def resolve_dependencies(stages: dict[str, PipelineStage], requested: list[str]) -> list[PipelineStage]:
    """
    Resolves user-requested stage dependencies. 

    Returns a list of Stages in the correct order of execution.
    """
    needed = set()
    stack = list(requested)

    while stack:

        stage_name = stack.pop() # final element
        assert stage_name in stages, f"{stage_name} is not in the supported stages; check typo?"

        if stage_name in needed:
            continue

        needed.add(stage_name)
        stack.extend(stages[stage_name].requires) # see PipelineStage dataclass
    
    # NOTE: this second part is Kahn's algorithm, which for a pipeline with a few stages is definitely overengineering; however, it is inexpensive and provides flexibility for future modules to be extended to Octavian, hence I'd argue in favour of using it
    depth = {n: 0 for n in needed}
    dependents = {n: [] for n in needed}

    for n in needed:

        for req in stages[n].requires:

            if req in needed:

                depth[n] += 1
                dependents[req].append(n)

    ready = [n for n in needed if depth[n] == 0] # top-level nodes
    ordered_stages = []

    while ready:

        n = ready.pop()
        ordered_stages.append(stages[n])

        for m in dependents[n]:
            depth[m] -= 1

            if depth[m] == 0:
                ready.append(m)

    assert len(ordered_stages) == len(needed), f"Stage dependencies are muddled, order unresolvable."

    return ordered_stages