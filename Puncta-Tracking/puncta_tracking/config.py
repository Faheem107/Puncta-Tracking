from pathlib import Path
import warnings

import yaml


# Module-level variables:
__external_paths__: dict[str, Path | None] = {
    "fiji-exe": None,
    "ilastik-exe": None,
    "trackmate-jython": None,
}

# Default configuration file
_default_config = Path(__file__).parent / ".defaults" / "config.yml"


def _set_path(ext_path_key: str, new_ext: str | Path | None) -> None:
    """
    Sets given path to given key in __external_paths__. If the
    given path does not exist, a FileNotFound error is raised.
    """
    if new_ext is None:
        __external_paths__[ext_path_key] = None
    else:
        new_ext_path = Path(new_ext)
        if not new_ext_path.exists():
            raise FileNotFoundError(f"Given file path {new_ext_path} does not exist")
        __external_paths__[ext_path_key] = new_ext_path


def set_fiji_path(fiji_path: str | Path | None) -> None:
    """
    Sets the path to the fiji executable to be used by the
    package, if the path exists. If the path does not exist,
    a FileNotFound error is raised.
    """
    _set_path("fiji-exe", fiji_path)


def get_fiji_path() -> Path | None:
    """
    Returns the current path for fiji executable
    """
    return __external_paths__["fiji-exe"]


def set_ilastik_path(ilastik_path: str | Path | None) -> None:
    """
    Sets the path to the ilastik executable to be used by the
    package, if the path exists. If the path does not exist,
    a FileNotFound error is raised.
    """
    _set_path("ilastik-exe", ilastik_path)


def get_ilastik_path() -> Path | None:
    """
    Returns the current value for ilastik executable
    """
    return __external_paths__["ilastik-exe"]


def set_trackmate_script_path(trackmate_path: str | Path | None) -> None:
    """
    Sets the path to the trackmate jython script to be used by the
    package, if the path exists. If the path does not exist,
    a FileNotFound error is raised.
    """
    _set_path("trackmate-jython", trackmate_path)


def get_trackmate_script_path() -> Path | None:
    """
    Returns the current value for trackmate jython script
    """
    return __external_paths__["trackmate-jython"]


def read_config(config_path: Path | str, warn: bool = True) -> None:
    """
    Reads a given configuration file (YAML) to update the paths
    for the fiji executable, ilastik executable and trackmate
    jython script.
    YAML script should have an entry "paths", with three
    subentries "fiji-exe", "ilastik-exe" and "trackmate-jython".
    Values of the entries will be set as the corresponding paths:
        fiji-exe -- fiji path
        ilastik-exe -- ilastik path
        trackmate-jython -- trackmate jython script path
    If a corresponding subentry is not found, __external_paths__
    is not changed. A warning is issued for each individual
    subentry not found. If there is no entry "paths", a separate
    warning is issued. Warnings can be suppressed by passing
    warn = False argument.
    """
    config_path = Path(config_path).absolute()
    with open(config_path, "r", encoding="utf-8") as file:
        config_data = yaml.safe_load(file)

    # Check if config_data is a dictionary. Loading a YAML
    # file can result in a dictionary, None or a scalar value
    try:
        config_data.items()
    except (AttributeError, TypeError):
        if warn:
            warnings.warn(f"No paths entry in {config_path}")
        return

    if "paths" in config_data:
        config_path_entries = config_data["paths"]
        path_setters = {
            "fiji-exe": set_fiji_path,
            "ilastik-exe": set_ilastik_path,
            "trackmate-jython": set_trackmate_script_path,
        }
        for path_entry, path_setter in path_setters.items():
            if path_entry in config_path_entries:
                path_setter(config_path_entries[path_entry])
            else:
                if warn:
                    warnings.warn(
                        f"No {path_entry} under paths specified in {config_path}"
                    )
    else:
        if warn:
            warnings.warn(f"No paths entry in {config_path}")
