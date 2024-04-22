import pkg_resources
import types

def get_imports():
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            # Split ensures you get root package, 
            # not just imported function
            name = val.__name__.split(".")[0]

        elif isinstance(val, type):
            name = val.__module__.split(".")[0]
            
        # Some packages are weird and have different
        # imported names vs. system/pip names. Unfortunately,
        # there is no systematic way to get pip names from
        # a package's imported name. You'll have to add
        # exceptions to this list manually!
        poorly_named_packages = {
            "PIL": "Pillow",
            "sklearn": "scikit-learn"
        }
        if name in poorly_named_packages.keys():
            name = poorly_named_packages[name]
            
        yield name

def update_requirements_file(file_path):
    imports = list(set(get_imports()))

    # The only way I found to get the version of the root package
    # from only the name of the package is to cross-check the names 
    # of installed packages vs. imported packages
    new_requirements = []
    for m in pkg_resources.working_set:
        if m.project_name in imports and m.project_name!="pip":
            new_requirements.append((m.project_name, m.version))

    # Read existing requirements from file
    existing_requirements = set()
    try:
        with open(file_path, 'r') as f:
            for line in f:
                existing_requirements.add(line.strip())
    except FileNotFoundError:
        pass

    # Append new requirements to existing ones
    updated_requirements = existing_requirements.union(set(f"{package}=={version}" for package, version in new_requirements))

    # Write updated requirements to file
    with open(file_path, 'w') as f:
        for requirement in sorted(updated_requirements):
            f.write(f"{requirement}\n")

# Example usage:
update_requirements_file("requirements.txt")
